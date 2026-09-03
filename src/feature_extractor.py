"""
Feature Extraction Module using Google MediaPipe Face Landmarker / FaceMesh.

Extracts 5 micro-expression and physiological spatial metrics per frame:
1. Eye Aspect Ratio (EAR) - Measures eye openness / blink dynamics.
2. Mouth Aspect Ratio (MAR) - Measures yawning activity.
3. Pitch - 3D head nodding (looking down / drowsiness nod).
4. Yaw - 3D head turning (looking left / right).
5. Roll - 3D head tilting (head dropping towards shoulder).
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calculate_ear(eye_coords: List[Tuple[int, int]]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for a single eye given 6 landmark points.
    Landmarks order: [P1(outer), P2(upper_left), P3(upper_right), P4(inner), P5(lower_right), P6(lower_left)]
    """
    v1 = calculate_distance(eye_coords[1], eye_coords[5])
    v2 = calculate_distance(eye_coords[2], eye_coords[4])
    h = calculate_distance(eye_coords[0], eye_coords[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def calculate_mar(mouth_coords: List[Tuple[int, int]]) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) given 8 landmark points around lips.
    Order: [P0(left corner), P1(upper_left), P2(upper_mid), P3(upper_right),
            P4(right corner), P5(lower_right), P6(lower_mid), P7(lower_left)]
    """
    v1_m = calculate_distance(mouth_coords[1], mouth_coords[7])
    v2_m = calculate_distance(mouth_coords[2], mouth_coords[6])
    v3_m = calculate_distance(mouth_coords[3], mouth_coords[5])
    h_m = calculate_distance(mouth_coords[0], mouth_coords[4])
    if h_m == 0:
        return 0.0
    return (v1_m + v2_m + v3_m) / (2.0 * h_m)


def calculate_head_pose(
    landmarks_2d: np.ndarray,
    frame_w: int,
    frame_h: int
) -> Tuple[float, float, float]:
    """
    Compute 3D Head Pose Euler angles (Pitch, Yaw, Roll) using solvePnP.

    Args:
        landmarks_2d: 6 selected facial 2D coordinates [[x, y], ...]
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Tuple of (pitch, yaw, roll) in degrees.
    """
    # Canonical 3D facial model points (in world/millimeter coordinates)
    face_3d = np.array([
        (0.0, 0.0, 0.0),            # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),       # Chin (landmark 152)
        (-225.0, 170.0, -135.0),    # Left eye left corner (landmark 33)
        (225.0, 170.0, -135.0),     # Right eye right corner (landmark 263)
        (-150.0, -150.0, -125.0),   # Left mouth corner (landmark 61)
        (150.0, -150.0, -125.0)     # Right mouth corner (landmark 291)
    ], dtype=np.float64)

    # Approximate camera intrinsic matrix (assuming principal point at center, no distortion)
    focal_length = float(frame_w)
    cam_matrix = np.array([
        [focal_length, 0, frame_w / 2.0],
        [0, focal_length, frame_h / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(face_3d, landmarks_2d, cam_matrix, dist_matrix)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = float(angles[0])  # Nodding (down/up)
    yaw = float(angles[1])    # Turning (left/right)
    roll = float(angles[2])   # Tilting (shoulder drop)

    return pitch, yaw, roll


def extract_facial_features_from_landmarks(
    landmarks,
    frame_w: int,
    frame_h: int
) -> Tuple[float, float, float, float, float]:
    """
    Extract 5 facial features from normalized landmarks.

    Args:
        landmarks: List/iterable of landmarks with .x and .y attributes in [0, 1].
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Tuple of (ear, mar, pitch, yaw, roll).
    """
    coords = [(int(pt.x * frame_w), int(pt.y * frame_h)) for pt in landmarks]

    # Landmark indices for Left and Right Eye
    left_eye = [coords[33], coords[160], coords[158], coords[133], coords[153], coords[144]]
    right_eye = [coords[362], coords[385], coords[387], coords[263], coords[373], coords[380]]

    ear_left = calculate_ear(left_eye)
    ear_right = calculate_ear(right_eye)
    ear = (ear_left + ear_right) / 2.0

    # Landmark indices for Mouth / Lips
    mouth = [
        coords[78], coords[81], coords[13], coords[311],
        coords[308], coords[402], coords[14], coords[178]
    ]
    mar = calculate_mar(mouth)

    # 6 Reference landmarks for Head Pose
    ref_indices = [1, 152, 33, 263, 61, 291]
    face_2d = np.array([coords[idx] for idx in ref_indices], dtype=np.float64)

    pitch, yaw, roll = calculate_head_pose(face_2d, frame_w, frame_h)

    return ear, mar, pitch, yaw, roll


class FacialLandmarkerPipeline:
    """
    Unified Landmarker Pipeline supporting both MediaPipe Tasks API
    and MediaPipe Solutions FaceMesh fallback.
    """

    def __init__(self, model_asset_path: Optional[str] = None):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Please install mediapipe: pip install mediapipe")

        self.mode = None
        self.detector = None

        # 1. Check or auto-download Tasks API model asset if path is provided or default
        target_model_path = model_asset_path
        if not target_model_path or not os.path.exists(target_model_path):
            default_dir = Path(__file__).resolve().parent.parent / "models"
            default_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = default_dir / "face_landmarker.task"
            
            if not candidate_path.exists():
                try:
                    import urllib.request
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    print(f"Downloading MediaPipe Face Landmarker model from {url}...")
                    urllib.request.urlretrieve(url, str(candidate_path))
                    print(f"✅ Downloaded face_landmarker.task to {candidate_path}")
                except Exception as dl_err:
                    print(f"Note: Could not auto-download model ({dl_err}).")

            if candidate_path.exists():
                target_model_path = str(candidate_path)

        # 2. Try Tasks API if model asset exists
        if target_model_path and os.path.exists(target_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=target_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=False,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                self.mode = "tasks_api"
            except Exception as e:
                print(f"Warning: Failed to load MediaPipe Tasks API ({e}). Falling back to FaceMesh solutions...")
                self.detector = None

        # 3. Fallback to MediaPipe Solutions FaceMesh
        if self.detector is None:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mode = "solutions_api"
            else:
                raise RuntimeError(
                    "No compatible MediaPipe FaceMesh or Tasks model found. "
                    "Provide a valid model_asset_path to 'face_landmarker.task' or ensure mediapipe solutions is supported."
                )

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        frame_w: int,
        frame_h: int
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Process a single RGB frame and extract 5 facial features.
        """
        if self.mode == "tasks_api":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.face_landmarks and len(results.face_landmarks) > 0:
                return extract_facial_features_from_landmarks(results.face_landmarks[0], frame_w, frame_h)
            return None

        elif self.mode == "solutions_api":
            results = self.detector.process(rgb_frame)
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                return extract_facial_features_from_landmarks(results.multi_face_landmarks[0].landmark, frame_w, frame_h)
            return None

        return None

    def close(self):
        """Release underlying landmarker resources."""
        if self.mode == "solutions_api" and self.detector:
            self.detector.close()
        elif self.mode == "tasks_api" and self.detector:
            self.detector.close()


def extract_features_from_video(
    video_path: Union[str, Path],
    pipeline: FacialLandmarkerPipeline,
    frame_skip: int = 5,
    max_frames: int = 5000,
    resize_dim: Tuple[int, int] = (640, 480),
    default_padding: Tuple[float, float, float, float, float] = (0.3, 0.0, 0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    Extract frame-by-frame 5 features (EAR, MAR, Pitch, Yaw, Roll) from a video file.
    Uses cap.grab() for fast non-blocking frame skipping.

    Args:
        video_path: Path to video file (.mp4, .mov, .avi, etc.)
        pipeline: Initialized FacialLandmarkerPipeline instance.
        frame_skip: Sample 1 frame every N frames.
        max_frames: Maximum sampled frames to extract per video.
        resize_dim: (width, height) to resize frames for inference.
        default_padding: 5-tuple default features when face is temporarily not detected.

    Returns:
        np.ndarray of shape (num_extracted_frames, 5) with float32 features.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video: {video_path}")
        return np.empty((0, 5), dtype=np.float32)

    features = []
    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < max_frames:
        if frame_count % frame_skip == 0:
            ret, frame = cap.read()
            if not ret:
                break

            if resize_dim:
                frame = cv2.resize(frame, resize_dim)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            feats = pipeline.process_frame(rgb_frame, frame_w=w, frame_h=h)
            if feats is not None:
                features.append(list(feats))
            else:
                features.append(list(default_padding))

            extracted_count += 1
        else:
            # Fast frame skipping without decoding pixel buffer
            ret = cap.grab()
            if not ret:
                break

        frame_count += 1

    cap.release()
    return np.array(features, dtype=np.float32)

