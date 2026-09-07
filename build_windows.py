"""Build a training-compatible temporal archive from frame-level caches."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import MAX_ALLOWED_PADDING_RATE
from src.dataset import UTARLDDDataset
from src.feature_cache import load_cache_metadata, validate_cached_arrays, window_cached_video


def parse_args():
    parser = argparse.ArgumentParser(description="Build temporal windows without rerunning MediaPipe.")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--frame_skip", type=int, required=True)
    parser.add_argument("--sequence_length", "--seq_length", dest="sequence_length", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)
    parser.add_argument("--max_padding_rate", type=float, default=MAX_ALLOWED_PADDING_RATE)
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("frame_skip", "sequence_length", "step_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if not 0.0 <= args.max_padding_rate <= 1.0:
        raise ValueError("--max_padding_rate must be between 0 and 1.")

    cache_dir = Path(args.cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Refusing to build windows from an incomplete cache.")
    entries = manifest.get("videos", [])
    if not entries:
        raise RuntimeError("Cache manifest contains no videos.")

    all_X = []
    all_masks = []
    all_y = []
    all_subjects = []
    all_folds = []
    all_video_ids = []
    all_start_frames = []
    all_end_frames = []
    all_start_sec = []
    all_end_sec = []
    video_keys = []
    skipped_short = []

    for video_index, entry in enumerate(entries):
        cache_path = cache_dir / entry["cache_file"]
        with np.load(cache_path, allow_pickle=False) as data:
            validate_cached_arrays(data)
            cached_metadata = load_cache_metadata(data)
            if cached_metadata["video_id"] != entry["video_id"]:
                raise RuntimeError(f"Manifest/cache video ID mismatch: {cache_path}")
            windowed = window_cached_video(
                data["features"],
                data["observed_mask"],
                data["frame_indices"],
                data["timestamps_sec"],
                frame_skip=args.frame_skip,
                sequence_length=args.sequence_length,
                step_size=args.step_size,
            )

        count = len(windowed["X"])
        video_keys.append(entry["video_id"])
        if count == 0:
            skipped_short.append(entry["video_id"])
            continue
        all_X.append(windowed["X"])
        all_masks.append(windowed["observed_mask"])
        all_y.append(np.full(count, entry["label"], dtype=np.int32))
        all_subjects.append(np.full(count, entry["subject_id"], dtype=np.int32))
        all_folds.append(np.full(count, entry["fold_id"], dtype=np.int32))
        all_video_ids.append(np.full(count, video_index, dtype=np.int32))
        all_start_frames.append(windowed["window_start_frame"])
        all_end_frames.append(windowed["window_end_frame"])
        all_start_sec.append(windowed["window_start_sec"])
        all_end_sec.append(windowed["window_end_sec"])

    if not all_X:
        raise RuntimeError("No video is long enough for this temporal configuration.")

    X = np.concatenate(all_X)
    observed_mask = np.concatenate(all_masks)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)
    folds = np.concatenate(all_folds)
    video_ids = np.concatenate(all_video_ids)
    dataset = UTARLDDDataset(X=X, y=y, folds=folds, subjects=subjects)
    integrity = dataset.validate()
    if integrity["padding_rate"] > args.max_padding_rate:
        raise RuntimeError(
            f"Missing-face padding rate {integrity['padding_rate']:.2%} exceeds "
            f"--max_padding_rate {args.max_padding_rate:.2%}."
        )

    build_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "cache_format_version": manifest.get("format_version"),
        "frame_skip": args.frame_skip,
        "sequence_length": args.sequence_length,
        "step_size": args.step_size,
        "num_windows": int(len(X)),
        "num_videos_with_windows": int(len(np.unique(video_ids))),
        "skipped_short_videos": skipped_short,
        "padding_rate": float(integrity["padding_rate"]),
        "fold_ids": dataset.available_folds,
    }

    output_path = Path(args.output_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("--output_path must end in .npz")
    if output_path.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=X,
            y=y,
            folds=folds,
            subjects=subjects,
            observed_mask=observed_mask,
            video_ids=video_ids,
            video_keys=np.asarray(video_keys),
            window_start_frame=np.concatenate(all_start_frames),
            window_end_frame=np.concatenate(all_end_frames),
            window_start_sec=np.concatenate(all_start_sec),
            window_end_sec=np.concatenate(all_end_sec),
            build_metadata_json=np.asarray(json.dumps(build_metadata, sort_keys=True)),
        )
    os.replace(temporary, output_path)

    print(f"Built {len(X)} windows with shape {X.shape}.")
    print(f"Subjects: {integrity['num_subjects']}; folds: {dataset.available_folds}")
    print(f"Missing-face padding rate: {integrity['padding_rate']:.2%}")
    if skipped_short:
        print(f"Warning: {len(skipped_short)} videos were shorter than the requested sequence.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
