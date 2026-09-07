"""Extract one reusable, frame-level UTA-RLDD feature cache per video."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import DEFAULT_DATASET_PATH, DEFAULT_LANDMARKER_MODEL_PATH
from src.dataset import find_all_video_files, parse_video_metadata
from src.feature_cache import (
    extract_frame_level_features,
    load_cache_metadata,
    new_manifest,
    save_cache_npz,
    source_signature,
    stable_video_id,
    validate_cached_arrays,
)
from src.feature_extractor import FacialLandmarkerPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frame-level EAR/MAR/head-pose caches for reusable temporal experiments."
    )
    parser.add_argument("--dataset_path", nargs="+", default=[DEFAULT_DATASET_PATH])
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--model_path", default=DEFAULT_LANDMARKER_MODEL_PATH)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="Debug only; omit for full extraction.")
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--resize_height", type=int, default=480)
    parser.add_argument(
        "--require_explicit_fold",
        action="store_true",
        help="Reject paths without Fold1..Fold5 instead of using the subject mapping fallback.",
    )
    parser.add_argument(
        "--allow_incomplete_dataset",
        action="store_true",
        help="Allow fewer than the canonical 180 videos/60 subjects (debug or partial data only).",
    )
    return parser.parse_args()


def _extract_one(task):
    video_path, output_path, model_path, use_gpu, max_frames, resize_dim, metadata = task
    pipeline = FacialLandmarkerPipeline(model_asset_path=model_path, use_gpu=use_gpu)
    try:
        arrays = extract_frame_level_features(
            video_path, pipeline, max_frames=max_frames, resize_dim=resize_dim
        )
    finally:
        pipeline.close()
    save_cache_npz(output_path, arrays, metadata)
    return {
        **metadata,
        "cache_file": str(Path("videos") / Path(output_path).name),
        "decoded_frames": int(len(arrays["features"])),
        "observed_frames": int(np.count_nonzero(arrays["observed_mask"])),
        "source_fps": float(arrays["source_fps"]),
        "reported_frame_count": int(arrays["reported_frame_count"]),
        "timestamp_source": str(arrays["timestamp_source"].item()),
    }


def _load_resumable(cache_path, expected_metadata):
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        validate_cached_arrays(data)
        actual = load_cache_metadata(data)
        for key in ("video_id", "subject_id", "fold_id", "label", "source", "extractor"):
            if actual.get(key) != expected_metadata.get(key):
                raise RuntimeError(
                    f"Existing cache conflicts with source metadata: {cache_path}. "
                    "Use a new cache directory for a different dataset."
                )
        return {
            **actual,
            "cache_file": str(Path("videos") / cache_path.name),
            "decoded_frames": int(len(data["features"])),
            "observed_frames": int(np.count_nonzero(data["observed_mask"])),
            "source_fps": float(data["source_fps"]) if "source_fps" in data.files else None,
            "reported_frame_count": int(data["reported_frame_count"]) if "reported_frame_count" in data.files else None,
            "timestamp_source": str(data["timestamp_source"].item()) if "timestamp_source" in data.files else None,
        }


def main():
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive.")
    if args.resize_width <= 0 or args.resize_height <= 0:
        raise ValueError("Resize dimensions must be positive.")
    if args.use_gpu and args.num_workers > 1:
        raise ValueError("Use --num_workers 1 with --use_gpu to avoid multiple GPU delegates.")

    cache_dir = Path(args.cache_dir)
    videos_dir = cache_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_paths = sorted({v for root in args.dataset_path for v in find_all_video_files(root)})
    if not video_paths:
        raise RuntimeError(f"No supported videos found below: {args.dataset_path}")

    # Initialize once before starting workers. This validates the runtime and lets
    # the existing pipeline download its Tasks model exactly once when necessary.
    bootstrap_pipeline = FacialLandmarkerPipeline(
        model_asset_path=args.model_path, use_gpu=args.use_gpu
    )
    effective_model_path = bootstrap_pipeline.model_asset_path or args.model_path
    backend_mode = bootstrap_pipeline.mode
    delegate_name = bootstrap_pipeline.delegate_name
    bootstrap_pipeline.close()

    resize_dim = (args.resize_width, args.resize_height)
    manifest = new_manifest(args.dataset_path, effective_model_path, resize_dim)
    manifest["backend_mode"] = backend_mode
    manifest["delegate_name"] = delegate_name
    tasks = []
    entries = []
    inferred_folds = []
    skipped_metadata = []

    for video_path in video_paths:
        parsed = parse_video_metadata(video_path)
        if parsed is None:
            skipped_metadata.append(video_path)
            continue
        if args.require_explicit_fold and parsed["fold_source"] != "path":
            raise RuntimeError(
                f"No explicit Fold1..Fold5 component in path: {video_path}. "
                "Remove --require_explicit_fold only after verifying the subject-to-fold mapping."
            )
        if parsed["fold_source"] != "path":
            inferred_folds.append(video_path)

        video_id = stable_video_id(video_path)
        metadata = {
            "video_id": video_id,
            "subject_id": int(parsed["subject_id"]),
            "fold_id": int(parsed["fold_id"]),
            "fold_source": parsed["fold_source"],
            "label": int(parsed["label"]),
            "filename": parsed["filename"],
            "source": source_signature(video_path),
            "extractor": {
                "model_sha256": manifest["model_sha256"],
                "backend_mode": backend_mode,
                "delegate_name": delegate_name,
                "resize_width": args.resize_width,
                "resize_height": args.resize_height,
                "max_frames": args.max_frames,
                "extraction_frame_skip": 1,
                "cache_format_version": manifest["format_version"],
            },
        }
        output_path = videos_dir / f"{video_id}.npz"
        resumed = _load_resumable(output_path, metadata)
        if resumed is not None:
            entries.append(resumed)
            continue
        tasks.append(
            (
                video_path,
                str(output_path),
                effective_model_path,
                args.use_gpu,
                args.max_frames,
                resize_dim,
                metadata,
            )
        )

    print(f"Found {len(video_paths)} videos; resuming {len(entries)}, extracting {len(tasks)}.")
    if skipped_metadata:
        print(f"Warning: {len(skipped_metadata)} files had no parseable label/subject and will be rejected.")
    if inferred_folds:
        print(
            f"Warning: {len(inferred_folds)} videos use the configured subject-to-fold mapping because "
            "their paths contain no explicit Fold1..Fold5 component."
        )

    failures = []
    if args.num_workers == 1:
        for task in tqdm(tasks, desc="Extracting videos"):
            try:
                entries.append(_extract_one(task))
            except Exception as exc:
                failures.append({"video_path": task[0], "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_task = {executor.submit(_extract_one, task): task for task in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Extracting videos"):
                task = future_to_task[future]
                try:
                    entries.append(future.result())
                except Exception as exc:
                    failures.append({"video_path": task[0], "error": repr(exc)})

    entries.sort(key=lambda item: (item["fold_id"], item["subject_id"], item["label"], item["video_id"]))
    subject_folds = {}
    for entry in entries:
        subject_folds.setdefault(entry["subject_id"], set()).add(entry["fold_id"])
    conflicts = {subject: sorted(folds) for subject, folds in subject_folds.items() if len(folds) != 1}
    if conflicts:
        failures.append({"error": f"Subjects assigned to multiple folds: {conflicts}"})
    if skipped_metadata:
        failures.append({"error": "Unparseable video metadata", "videos": skipped_metadata})

    subject_labels = {}
    fold_subjects = {}
    for entry in entries:
        subject_labels.setdefault(entry["subject_id"], []).append(entry["label"])
        fold_subjects.setdefault(entry["fold_id"], set()).add(entry["subject_id"])
    invalid_subject_classes = {
        subject: labels
        for subject, labels in subject_labels.items()
        if sorted(labels) != [0, 1, 2]
    }
    canonical_errors = []
    if len(entries) != 180:
        canonical_errors.append(f"expected 180 videos, found {len(entries)}")
    if len(subject_folds) != 60:
        canonical_errors.append(f"expected 60 subjects, found {len(subject_folds)}")
    if invalid_subject_classes:
        canonical_errors.append(
            f"subjects without exactly one video for labels [0, 1, 2]: {invalid_subject_classes}"
        )
    fold_counts = {fold: len(subjects) for fold, subjects in sorted(fold_subjects.items())}
    if fold_counts != {1: 12, 2: 12, 3: 12, 4: 12, 5: 12}:
        canonical_errors.append(f"expected 12 subjects in each fold, found {fold_counts}")
    if canonical_errors and not args.allow_incomplete_dataset:
        failures.append({"error": "Non-canonical UTA-RLDD inventory", "details": canonical_errors})
    elif canonical_errors:
        print("Warning: incomplete/non-canonical dataset explicitly allowed:")
        for error in canonical_errors:
            print(f"  - {error}")

    manifest["videos"] = entries
    manifest["num_videos"] = len(entries)
    manifest["num_subjects"] = len(subject_folds)
    manifest["fold_subject_counts"] = fold_counts
    manifest["canonical_inventory_errors"] = canonical_errors
    manifest["failures"] = failures
    manifest["status"] = "complete" if not failures and len(entries) == len(video_paths) else "failed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if manifest["status"] != "complete":
        raise RuntimeError(
            f"Cache is incomplete ({len(failures)} failure records). Inspect {cache_dir / 'manifest.json'}."
        )

    total_frames = sum(entry["decoded_frames"] for entry in entries)
    observed_frames = sum(entry["observed_frames"] for entry in entries)
    print(f"Cache complete: {len(entries)} videos, {len(subject_folds)} subjects, {total_frames} frames.")
    print(f"Face detection coverage: {observed_frames / total_frames:.2%}")
    print(f"Manifest: {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed per-video files can be resumed by running the same command.")
        sys.exit(130)
