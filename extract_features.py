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
    LABEL_NAMES,
    MAX_ALLOWED_PADDING_RATE,
    TOTAL_FOLDS,
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
        nargs="+",
        type=str,
        default=[DEFAULT_DATASET_PATH],
        help=(
            "One or more root directories containing UTA-RLDD videos. Multiple roots are "
            "useful on Kaggle when Fold 5 is mounted as a separate dataset."
        )
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
        default=None,
        help="Optional max sampled frames per video (default: None = Full Video)."
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
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Attempt GPU delegate for MediaPipe inference."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for video extraction (default: 4)."
    )
    parser.add_argument(
        "--max_padding_rate",
        type=float,
        default=MAX_ALLOWED_PADDING_RATE,
        help=(
            "Abort without saving when the missing-face padding rate exceeds this value "
            f"(default: {MAX_ALLOWED_PADDING_RATE:.2f})."
        )
    )
    return parser.parse_args()


def _process_video_worker(task):
    video_path, model_path, use_gpu, frame_skip, max_frames, seq_length, step_size = task
    meta = parse_video_metadata(video_path)
    if meta is None:
        return None

    pipeline = FacialLandmarkerPipeline(model_asset_path=model_path, use_gpu=use_gpu)
    try:
        video_feats = extract_features_from_video(
            video_path=video_path,
            pipeline=pipeline,
            frame_skip=frame_skip,
            max_frames=max_frames
        )
    finally:
        pipeline.close()

    if len(video_feats) < seq_length:
        return None

    return create_sliding_windows_for_video(
        video_features=video_feats,
        label=meta["label"],
        subject_id=meta["subject_id"],
        fold_id=meta["fold_id"],
        seq_length=seq_length,
        step_size=step_size
    )


def main():
    args = parse_args()
    set_seed(42)

    if not 0.0 <= args.max_padding_rate <= 1.0:
        raise ValueError("--max_padding_rate must be between 0 and 1.")

    print("=" * 70)
    print("UTA-RLDD ZERO-LEAKAGE FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Dataset Paths: {args.dataset_path}")
    print(f"Output File  : {args.output_path}")
    print(f"Frame Skip   : {args.frame_skip} (Sample every {args.frame_skip}th frame)")
    print(f"Max Frames   : {args.max_frames if args.max_frames else 'Full Video (Unlimited)'}")
    print(f"Seq Length   : {args.seq_length} timesteps")
    print(f"Step Size    : {args.step_size} stride")
    print(f"Workers      : {args.num_workers} parallel processes")
    print(f"GPU Delegate : {args.use_gpu}")
    print("=" * 70)

    # 1. Discover all videos
    video_files = sorted({
        video
        for dataset_path in args.dataset_path
        for video in find_all_video_files(dataset_path)
    })
    if not video_files:
        print(f"❌ Error: No video files found in {args.dataset_path}")
        print("Please check the dataset path and try again.")
        sys.exit(1)

    print(f"Found {len(video_files)} video files in dataset.\n")

    # 2. Process videos in parallel or serial
    all_X_windows, all_y_windows, all_sub_windows, all_fold_windows = [], [], [], []
    processed_videos = 0
    skipped_videos = 0

    tasks = [
        (v, args.model_path, args.use_gpu, args.frame_skip, args.max_frames, args.seq_length, args.step_size)
        for v in video_files
    ]

    print("Starting parallel video processing & feature extraction:")
    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_video = {executor.submit(_process_video_worker, task): task[0] for task in tasks}
            pbar = tqdm(as_completed(future_to_video), total=len(tasks), desc="Extracting features")
            for future in pbar:
                vid_path = future_to_video[future]
                try:
                    result = future.result()
                    if result is not None:
                        X_win, y_win, sub_win, fold_win = result
                        all_X_windows.append(X_win)
                        all_y_windows.append(y_win)
                        all_sub_windows.append(sub_win)
                        all_fold_windows.append(fold_win)
                        processed_videos += 1
                        pbar.set_postfix({"last_vid": Path(vid_path).name, "processed": processed_videos})
                    else:
                        skipped_videos += 1
                except Exception as exc:
                    print(f"\nWarning: Exception on video {vid_path}: {exc}")
                    skipped_videos += 1
    else:
        # Serial execution
        pipeline = FacialLandmarkerPipeline(model_asset_path=args.model_path, use_gpu=args.use_gpu)
        pbar = tqdm(video_files, desc="Extracting features")
        for video_path in pbar:
            meta = parse_video_metadata(video_path)
            if meta is None:
                skipped_videos += 1
                continue

            label = meta["label"]
            subject_id = meta["subject_id"]
            fold_id = meta["fold_id"]
            pbar.set_postfix({"vid": Path(video_path).name, "sub": subject_id, "processed": processed_videos})

            video_feats = extract_features_from_video(
                video_path=video_path,
                pipeline=pipeline,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames
            )

            if len(video_feats) < args.seq_length:
                continue

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

    # Reject broken extraction output before an expensive training run can use it.
    integrity = dataset.validate()
    padding_rate = float(integrity["padding_rate"])
    if padding_rate > args.max_padding_rate:
        raise RuntimeError(
            f"Extraction aborted: missing-face padding rate is {padding_rate:.2%}, above "
            f"the configured limit of {args.max_padding_rate:.2%}. No NPZ was saved."
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

    print(f"\nMissing-face padding rate: {padding_rate:.2%}")
    for warning in integrity["warnings"]:
        print(f"Warning: {warning}")

    print("\nFold Distribution:")
    for f_idx in dataset.available_folds:
        count = int(np.sum(dataset.folds == f_idx))
        print(f"  - Fold {f_idx} : {count:>6} samples")

    missing_folds = sorted(set(range(1, TOTAL_FOLDS + 1)) - set(dataset.available_folds))
    if missing_folds:
        print(
            f"Warning: missing folds {missing_folds}. The saved archive is valid for "
            f"{len(dataset.available_folds)}-fold evaluation, not {TOTAL_FOLDS}-fold evaluation."
        )

    print(f"\nSaved .npz dataset to: {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
