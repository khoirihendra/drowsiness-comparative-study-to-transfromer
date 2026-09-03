"""
Standalone Feature Extraction Script for UTA Real-Life Drowsiness Dataset.

Extracts 5 facial physiological and head pose metrics:
(EAR, MAR, Pitch, Yaw, Roll) using MediaPipe, and generates
zero-leakage temporal sequences tagged with Fold and Subject metadata.

Usage:
    python extract_features.py --dataset_path /path/to/uta-rldd --output_path output/extracted_features/uta_rldd_5features.npz
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LANDMARKER_MODEL_PATH,
    FRAME_SKIP,
    MAX_FRAMES_PER_VIDEO,
    SEQ_LENGTH,
    STEP_SIZE,
    FEATURES_DIR,
    LABEL_NAMES
)
from src.utils import set_seed
from src.feature_extractor import FacialLandmarkerPipeline, extract_features_from_video
from src.dataset import (
    find_all_video_files,
    parse_video_metadata,
    create_sliding_windows_for_video,
    UTARLDDDataset
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract EAR, MAR, and Head Pose features from UTA-RLDD videos."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Root directory containing UTA-RLDD video files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(FEATURES_DIR / "uta_rldd_features_seq30.npz"),
        help="Destination path for output .npz dataset file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_LANDMARKER_MODEL_PATH,
        help="Path to face_landmarker.task model file."
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=FRAME_SKIP,
        help="Extract 1 frame every N frames (default: 5)."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=MAX_FRAMES_PER_VIDEO,
        help="Maximum extracted frames per video (default: 5000)."
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=SEQ_LENGTH,
        help="Temporal sliding window sequence length (default: 30)."
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=STEP_SIZE,
        help="Sliding window step stride (default: 1)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    print("=" * 70)
    print("UTA-RLDD ZERO-LEAKAGE FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Dataset Path : {args.dataset_path}")
    print(f"Output File  : {args.output_path}")
    print(f"Frame Skip   : {args.frame_skip} (Sample every {args.frame_skip}th frame)")
    print(f"Max Frames   : {args.max_frames} per video")
    print(f"Seq Length   : {args.seq_length} timesteps")
    print(f"Step Size    : {args.step_size} stride")
    print("=" * 70)

    # 1. Discover all videos
    video_files = find_all_video_files(args.dataset_path)
    if not video_files:
        print(f"❌ Error: No video files found in {args.dataset_path}")
        print("Please check the dataset path and try again.")
        sys.exit(1)

    print(f"Found {len(video_files)} video files in dataset.\n")

    # 2. Initialize MediaPipe Landmarker
    print("Initializing MediaPipe Face Landmarker...")
    try:
        pipeline = FacialLandmarkerPipeline(model_asset_path=args.model_path)
        print(f"✅ Landmarker initialized successfully (Mode: {pipeline.mode})!\n")
    except Exception as e:
        print(f"❌ Error initializing landmarker: {e}")
        sys.exit(1)

    # 3. Process each video individually (Zero Cross-Video Boundary Leakage)
    all_X_windows = []
    all_y_windows = []
    all_sub_windows = []
    all_fold_windows = []

    processed_videos = 0
    skipped_videos = 0

    print("Starting video processing & feature extraction:")
    for video_path in tqdm(video_files, desc="Extracting features"):
        meta = parse_video_metadata(video_path)
        if meta is None:
            skipped_videos += 1
            continue

        label = meta["label"]
        subject_id = meta["subject_id"]
        fold_id = meta["fold_id"]

        # Extract frame-by-frame 5 features (EAR, MAR, Pitch, Yaw, Roll)
        video_feats = extract_features_from_video(
            video_path=video_path,
            pipeline=pipeline,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )

        if len(video_feats) < args.seq_length:
            continue

        # Form temporal sliding windows strictly for this single video
        X_win, y_win, sub_win, fold_win = create_sliding_windows_for_video(
            video_features=video_feats,
            label=label,
            subject_id=subject_id,
            fold_id=fold_id,
            seq_length=args.seq_length,
            step_size=args.step_size
        )

        all_X_windows.append(X_win)
        all_y_windows.append(y_win)
        all_sub_windows.append(sub_win)
        all_fold_windows.append(fold_win)
        processed_videos += 1

    pipeline.close()

    if not all_X_windows:
        print("❌ Error: No valid feature sequences could be extracted.")
        sys.exit(1)

    # 4. Concatenate into dataset container
    X_data = np.concatenate(all_X_windows, axis=0)
    y_data = np.concatenate(all_y_windows, axis=0)
    sub_data = np.concatenate(all_sub_windows, axis=0)
    fold_data = np.concatenate(all_fold_windows, axis=0)

    dataset = UTARLDDDataset(
        X=X_data,
        y=y_data,
        folds=fold_data,
        subjects=sub_data
    )

    # 5. Save to disk
    dataset.save_npz(args.output_path)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Processed Videos : {processed_videos} (Skipped: {skipped_videos})")
    print(f"Total Sequences  : {len(dataset)} samples")
    print(f"Feature Shape    : {dataset.X.shape} (Samples, Timesteps, 5 Features)")
    print(f"Label Shape      : {dataset.y.shape}")
    print("\nClass Distribution:")
    for c_idx, c_name in enumerate(LABEL_NAMES):
        count = int(np.sum(dataset.y == c_idx))
        pct = (count / len(dataset)) * 100
        print(f"  - [{c_idx}] {c_name:<14}: {count:>6} samples ({pct:.1f}%)")

    print("\nFold Distribution:")
    for f_idx in range(1, 6):
        count = int(np.sum(dataset.folds == f_idx))
        print(f"  - Fold {f_idx} : {count:>6} samples")

    print(f"\nSaved .npz dataset to: {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
