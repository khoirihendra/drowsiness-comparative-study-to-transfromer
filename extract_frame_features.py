"""Extract reusable frame-level facial features one UTA-RLDD fold at a time.

Completed videos are checkpointed individually.  If Kaggle interrupts a run,
run the same command again and valid checkpoints will be reused.
"""

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LANDMARKER_MODEL_PATH,
    FRAME_SKIP,
    MAX_ALLOWED_PADDING_RATE,
)
from src.dataset import find_all_video_files, parse_video_metadata
from src.feature_extractor import FacialLandmarkerPipeline, extract_features_from_video
from src.frame_dataset import frame_padding_rate, save_frame_feature_shard
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract reusable frame-level features for one UTA-RLDD fold."
    )
    parser.add_argument("--dataset_path", nargs="+", default=[DEFAULT_DATASET_PATH])
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        "--output_dir",
        default="output/frame_features",
        help="Root directory for fold archives and resumable per-video checkpoints.",
    )
    parser.add_argument("--model_path", default=DEFAULT_LANDMARKER_MODEL_PATH)
    parser.add_argument("--frame_skip", type=int, default=FRAME_SKIP)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional sampled-frame limit per video; omit to process complete videos.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--max_padding_rate", type=float, default=MAX_ALLOWED_PADDING_RATE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore and overwrite existing per-video checkpoints and the fold archive.",
    )
    return parser.parse_args()


def checkpoint_path(cache_dir: Path, video_path: str, metadata) -> Path:
    normalized = str(Path(video_path)).replace("\\", "/").lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return cache_dir / (
        f"subject_{metadata['subject_id']:02d}_label_{metadata['label']}_{Path(video_path).stem}_{digest}.npz"
    )


def save_checkpoint(path: Path, features: np.ndarray, video_path: str, metadata, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".partial")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            features=np.asarray(features, dtype=np.float32),
            video_path=np.asarray(video_path),
            label=np.asarray(metadata["label"], dtype=np.int32),
            subject_id=np.asarray(metadata["subject_id"], dtype=np.int32),
            fold_id=np.asarray(metadata["fold_id"], dtype=np.int32),
            frame_skip=np.asarray(args.frame_skip, dtype=np.int32),
            max_frames=np.asarray(-1 if args.max_frames is None else args.max_frames, dtype=np.int64),
        )
    temporary_path.replace(path)


def load_checkpoint(path: Path, video_path: str, metadata, args):
    if not path.exists() or args.force:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_max = -1 if args.max_frames is None else args.max_frames
            matches = (
                str(data["video_path"]) == video_path
                and int(data["label"]) == metadata["label"]
                and int(data["subject_id"]) == metadata["subject_id"]
                and int(data["fold_id"]) == metadata["fold_id"]
                and int(data["frame_skip"]) == args.frame_skip
                and int(data["max_frames"]) == expected_max
            )
            features = data["features"]
            if matches and features.ndim == 2 and features.shape[1] == 5:
                return features
    except (OSError, ValueError, KeyError):
        pass
    return None


def process_video(task):
    video_path, metadata, args_dict = task
    pipeline = FacialLandmarkerPipeline(
        model_asset_path=args_dict["model_path"], use_gpu=args_dict["use_gpu"]
    )
    try:
        features = extract_features_from_video(
            video_path,
            pipeline,
            frame_skip=args_dict["frame_skip"],
            max_frames=args_dict["max_frames"],
        )
    finally:
        pipeline.close()
    return video_path, metadata, features


def record_from_checkpoint(path: Path, video_path: str, metadata, args):
    features = load_checkpoint(path, video_path, metadata, args)
    if features is None:
        return None
    return {
        "features": features,
        "video_path": video_path,
        "label": metadata["label"],
        "subject_id": metadata["subject_id"],
        "fold_id": metadata["fold_id"],
    }


def main():
    args = parse_args()
    set_seed(42)
    if args.frame_skip <= 0:
        raise ValueError("--frame_skip must be greater than zero.")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be greater than zero.")
    if not 0.0 <= args.max_padding_rate <= 1.0:
        raise ValueError("--max_padding_rate must be between 0 and 1.")

    candidates = sorted(
        {
            video
            for root in args.dataset_path
            for video in find_all_video_files(root)
        }
    )
    selected = []
    for video_path in candidates:
        metadata = parse_video_metadata(video_path)
        if metadata is not None and metadata["fold_id"] == args.fold:
            selected.append((video_path, metadata))
    if not selected:
        raise RuntimeError(f"No valid videos belonging to fold {args.fold} were found.")

    fold_dir = Path(args.output_dir) / f"fold_{args.fold}"
    cache_dir = fold_dir / "video_checkpoints"
    output_path = fold_dir / f"frame_features_fold_{args.fold}.npz"
    if output_path.exists() and not args.force:
        print(f"Fold archive already exists: {output_path}")
        print("Use --force only if you intend to re-extract it.")
        return

    print(f"Fold {args.fold}: {len(selected)} videos")
    print(f"Output: {output_path}")
    records_by_video = {}
    pending = []
    for video_path, metadata in selected:
        cache_path = checkpoint_path(cache_dir, video_path, metadata)
        record = record_from_checkpoint(cache_path, video_path, metadata, args)
        if record is None:
            pending.append((video_path, metadata))
        else:
            records_by_video[video_path] = record
    print(f"Resuming from {len(records_by_video)} checkpoints; {len(pending)} videos remain.")

    failures = []
    if args.num_workers > 1 and pending:
        args_dict = {
            "model_path": args.model_path,
            "use_gpu": args.use_gpu,
            "frame_skip": args.frame_skip,
            "max_frames": args.max_frames,
        }
        tasks = [(video, metadata, args_dict) for video, metadata in pending]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_video, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Videos"):
                video_path, metadata = futures.pop(future)
                try:
                    _, _, features = future.result()
                    if len(features) == 0:
                        raise RuntimeError("video produced zero sampled frames")
                    cache_path = checkpoint_path(cache_dir, video_path, metadata)
                    save_checkpoint(cache_path, features, video_path, metadata, args)
                    records_by_video[video_path] = record_from_checkpoint(
                        cache_path, video_path, metadata, args
                    )
                    del features
                except Exception as exc:
                    failures.append((video_path, str(exc)))
    elif pending:
        pipeline = FacialLandmarkerPipeline(args.model_path, use_gpu=args.use_gpu)
        try:
            for video_path, metadata in tqdm(pending, desc="Videos"):
                try:
                    features = extract_features_from_video(
                        video_path,
                        pipeline,
                        frame_skip=args.frame_skip,
                        max_frames=args.max_frames,
                    )
                    if len(features) == 0:
                        raise RuntimeError("video produced zero sampled frames")
                    cache_path = checkpoint_path(cache_dir, video_path, metadata)
                    save_checkpoint(cache_path, features, video_path, metadata, args)
                    records_by_video[video_path] = record_from_checkpoint(
                        cache_path, video_path, metadata, args
                    )
                except Exception as exc:
                    failures.append((video_path, str(exc)))
        finally:
            pipeline.close()

    if failures:
        print("Failed videos:")
        for video_path, error in failures:
            print(f"  {video_path}: {error}")
        raise RuntimeError(
            f"Fold {args.fold} is incomplete ({len(failures)} failures). Fix the errors and rerun; "
            "completed video checkpoints will be reused."
        )

    ordered_records = [records_by_video[video] for video, _ in selected]
    all_features = np.concatenate([record["features"] for record in ordered_records], axis=0)
    padding_rate = frame_padding_rate(all_features)
    if padding_rate > args.max_padding_rate:
        raise RuntimeError(
            f"Fold archive not saved: missing-face padding rate {padding_rate:.2%} exceeds "
            f"the configured limit {args.max_padding_rate:.2%}."
        )
    if np.all(np.ptp(all_features, axis=0) == 0):
        raise RuntimeError("Fold archive not saved: every extracted feature is constant.")

    save_frame_feature_shard(output_path, ordered_records, frame_skip=args.frame_skip)
    print(f"Saved {len(ordered_records)} videos / {len(all_features)} sampled frames.")
    print(f"Missing-face padding rate: {padding_rate:.2%}")
    print(f"Fold archive: {output_path}")


if __name__ == "__main__":
    main()
