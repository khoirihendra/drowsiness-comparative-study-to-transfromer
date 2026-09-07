"""Reusable frame-level feature cache and temporal window construction.

The expensive MediaPipe pass is deliberately independent of frame sampling and
window configuration.  Each cached file represents exactly one source video.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:  # Window construction does not require OpenCV.
    cv2 = None

from config import FEATURE_NAMES, MISSING_FEATURE_VECTOR, TOTAL_RAW_FEATURES


CACHE_FORMAT_VERSION = 1


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_video_id(video_path: Union[str, Path]) -> str:
    """Create a filesystem-safe ID that stays stable for a given source path."""
    path = Path(video_path).resolve()
    slug = "_".join(part for part in path.stem.lower().split() if part) or "video"
    slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in slug)
    path_digest = hashlib.sha256(str(path).casefold().encode("utf-8")).hexdigest()[:16]
    return f"{slug[:40]}_{path_digest}"


def source_signature(video_path: Union[str, Path]) -> Dict[str, Union[str, int]]:
    """Cheap source identity used to detect changed videos during resume."""
    path = Path(video_path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _timestamp_for_frame(cap, frame_index: int, fps: float) -> Tuple[float, str]:
    pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    if np.isfinite(pos_msec) and (pos_msec > 0.0 or frame_index == 0):
        return pos_msec / 1000.0, "decoder"
    if np.isfinite(fps) and fps > 0.0:
        return frame_index / fps, "nominal_fps"
    return float(frame_index), "frame_index"


def extract_frame_level_features(
    video_path: Union[str, Path],
    pipeline,
    max_frames: Optional[int] = None,
    resize_dim: Optional[Tuple[int, int]] = (640, 480),
) -> Dict[str, np.ndarray]:
    """Extract every decoded frame and retain timestamps plus detection status.

    Missing detections are stored as NaN instead of a physiological-looking
    constant.  No labels, fold statistics, calibration, or imputation are used.
    """
    if cv2 is None:
        raise ImportError("OpenCV is required for video extraction: pip install opencv-python")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features = []
    observed = []
    frame_indices = []
    timestamps = []
    timestamp_sources = set()
    frame_index = 0

    try:
        while max_frames is None or max_frames <= 0 or frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp, timestamp_source = _timestamp_for_frame(cap, frame_index, fps)
            timestamp_sources.add(timestamp_source)
            if resize_dim is not None:
                frame = cv2.resize(frame, resize_dim)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            values = pipeline.process_frame(rgb_frame, frame_w=width, frame_h=height)

            if values is None:
                features.append(np.full(TOTAL_RAW_FEATURES, np.nan, dtype=np.float32))
                observed.append(False)
            else:
                values_array = np.asarray(values, dtype=np.float32)
                if values_array.shape != (TOTAL_RAW_FEATURES,) or not np.isfinite(values_array).all():
                    features.append(np.full(TOTAL_RAW_FEATURES, np.nan, dtype=np.float32))
                    observed.append(False)
                else:
                    features.append(values_array)
                    observed.append(True)

            frame_indices.append(frame_index)
            timestamps.append(timestamp)
            frame_index += 1
    finally:
        cap.release()

    if not features:
        raise RuntimeError(f"No frames could be decoded from: {video_path}")

    timestamp_source = (
        next(iter(timestamp_sources)) if len(timestamp_sources) == 1 else "mixed_decoder_nominal"
    )
    return {
        "features": np.asarray(features, dtype=np.float32),
        "observed_mask": np.asarray(observed, dtype=np.bool_),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "timestamps_sec": np.asarray(timestamps, dtype=np.float64),
        "source_fps": np.asarray(fps, dtype=np.float64),
        "reported_frame_count": np.asarray(reported_frame_count, dtype=np.int64),
        "timestamp_source": np.asarray(timestamp_source),
    }


def validate_cached_arrays(data) -> None:
    """Reject incomplete or internally inconsistent per-video cache files."""
    required = {"features", "observed_mask", "frame_indices", "timestamps_sec"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Cache file is missing arrays: {sorted(missing)}")

    features = data["features"]
    observed = data["observed_mask"]
    frame_indices = data["frame_indices"]
    timestamps = data["timestamps_sec"]
    lengths = {len(features), len(observed), len(frame_indices), len(timestamps)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Cache arrays must have the same non-zero length.")
    if features.ndim != 2 or features.shape[1] != TOTAL_RAW_FEATURES:
        raise ValueError(f"Expected features shaped (frames, {TOTAL_RAW_FEATURES}), got {features.shape}.")
    if not np.all(np.diff(frame_indices) > 0):
        raise ValueError("Cached frame indices must be strictly increasing.")
    if not np.all(np.diff(timestamps) >= 0):
        raise ValueError("Cached timestamps must be monotonic.")
    if np.any(observed & ~np.isfinite(features).all(axis=1)):
        raise ValueError("Observed cache rows must contain five finite features.")


def save_cache_npz(output_path: Union[str, Path], arrays: Dict[str, np.ndarray], metadata: Dict) -> None:
    """Atomically save a single-video cache."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, output)


def load_cache_metadata(data) -> Dict:
    if "metadata_json" not in data.files:
        raise ValueError("Cache file has no metadata_json array.")
    return json.loads(str(data["metadata_json"].item()))


def window_cached_video(
    features: np.ndarray,
    observed_mask: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    frame_skip: int,
    sequence_length: int,
    step_size: int,
) -> Dict[str, np.ndarray]:
    """Subsample a cached video and form windows without crossing its boundary."""
    for name, value in (
        ("frame_skip", frame_skip),
        ("sequence_length", sequence_length),
        ("step_size", step_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}.")

    features = np.asarray(features, dtype=np.float32)
    observed_mask = np.asarray(observed_mask, dtype=np.bool_)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float64)
    if not (len(features) == len(observed_mask) == len(frame_indices) == len(timestamps_sec)):
        raise ValueError("Cached arrays have inconsistent lengths.")

    sample_positions = np.flatnonzero(frame_indices % frame_skip == 0)
    sampled_features = features[sample_positions]
    sampled_observed = observed_mask[sample_positions]
    sampled_frames = frame_indices[sample_positions]
    sampled_timestamps = timestamps_sec[sample_positions]

    if len(sampled_features) < sequence_length:
        empty_features = np.empty((0, sequence_length, TOTAL_RAW_FEATURES), dtype=np.float32)
        empty_mask = np.empty((0, sequence_length), dtype=np.bool_)
        return {
            "X": empty_features,
            "observed_mask": empty_mask,
            "window_start_frame": np.empty(0, dtype=np.int64),
            "window_end_frame": np.empty(0, dtype=np.int64),
            "window_start_sec": np.empty(0, dtype=np.float64),
            "window_end_sec": np.empty(0, dtype=np.float64),
        }

    starts = np.arange(0, len(sampled_features) - sequence_length + 1, step_size, dtype=np.int64)
    X = np.empty((len(starts), sequence_length, TOTAL_RAW_FEATURES), dtype=np.float32)
    masks = np.empty((len(starts), sequence_length), dtype=np.bool_)
    for output_index, start in enumerate(starts):
        stop = start + sequence_length
        X[output_index] = sampled_features[start:stop]
        masks[output_index] = sampled_observed[start:stop]

    missing = ~masks
    if np.any(missing):
        X[missing] = np.asarray(MISSING_FEATURE_VECTOR, dtype=np.float32)

    ends = starts + sequence_length - 1
    return {
        "X": X,
        "observed_mask": masks,
        "window_start_frame": sampled_frames[starts],
        "window_end_frame": sampled_frames[ends],
        "window_start_sec": sampled_timestamps[starts],
        "window_end_sec": sampled_timestamps[ends],
    }


def new_manifest(dataset_roots, model_path, resize_dim) -> Dict:
    model = Path(model_path).resolve() if model_path else None
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "status": "in_progress",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_roots": [str(Path(root).resolve()) for root in dataset_roots],
        "feature_names": list(FEATURE_NAMES),
        "extraction_frame_skip": 1,
        "resize_width": resize_dim[0] if resize_dim else None,
        "resize_height": resize_dim[1] if resize_dim else None,
        "model_path": str(model) if model else None,
        "model_sha256": sha256_file(model) if model and model.exists() else None,
        "videos": [],
    }
