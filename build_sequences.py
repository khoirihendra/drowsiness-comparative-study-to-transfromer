"""Build configurable temporal windows from frame-feature fold archives."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SEQ_LENGTH, STEP_SIZE, TOTAL_RAW_FEATURES
from src.dataset import UTARLDDDataset
from src.frame_dataset import (
    build_npy_sequence_dataset,
    discover_frame_feature_shards,
    iter_videos,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create temporal sequences without rerunning MediaPipe extraction."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Fold .npz archives or directories containing frame_features_fold_*.npz.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_length", type=int, default=SEQ_LENGTH)
    parser.add_argument("--step_size", type=int, default=STEP_SIZE)
    parser.add_argument(
        "--format",
        choices=("npy", "npz", "both"),
        default="npy",
        help="npy creates X.npy, Y.npy, subject.npy, folds.npy; npz matches this repo's trainer.",
    )
    parser.add_argument(
        "--npz_name",
        default=None,
        help="NPZ filename (default: uta_rldd_features_seq<length>_step<stride>.npz).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seq_length <= 0 or args.step_size <= 0:
        raise ValueError("--seq_length and --step_size must be greater than zero.")

    shards = discover_frame_feature_shards(args.input)
    videos = list(iter_videos(shards))
    frame_skips = set()
    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as shard:
            frame_skips.add(int(shard["frame_skip"]))
    if len(frame_skips) != 1:
        raise ValueError(f"All folds must use the same frame_skip; found {sorted(frame_skips)}.")

    output_dir = Path(args.output_dir)
    npz_path = None
    if args.format in ("npz", "both"):
        name = args.npz_name or (
            f"uta_rldd_features_seq{args.seq_length}_step{args.step_size}.npz"
        )
        npz_path = output_dir / name
        if npz_path.exists() and not args.overwrite:
            raise FileExistsError(f"NPZ output already exists: {npz_path}")

    paths, total_windows = build_npy_sequence_dataset(
        videos, output_dir, args.seq_length, args.step_size, args.overwrite
    )

    # Validate the exact disk-backed result. This also catches fold leakage and
    # broken/constant extraction before training starts.
    dataset = UTARLDDDataset(
        X=np.load(paths["X"], mmap_mode="r"),
        y=np.load(paths["y"], mmap_mode="r"),
        subjects=np.load(paths["subjects"], mmap_mode="r"),
        folds=np.load(paths["folds"], mmap_mode="r"),
        copy=False,
    )
    summary = dataset.validate()

    if args.format in ("npz", "both"):
        dataset.save_npz(npz_path)

    if args.format == "npz":
        dataset.close()
        for path in paths.values():
            path.unlink()

    print(f"Built {total_windows} windows from {len(videos)} videos and {len(shards)} fold shards.")
    print(f"Shape: ({total_windows}, {args.seq_length}, {TOTAL_RAW_FEATURES})")
    print(f"Folds: {summary['available_folds']}; subjects: {summary['num_subjects']}")
    if args.format in ("npy", "both"):
        print(f"Four-array dataset: {output_dir}")
    if npz_path is not None:
        print(f"Trainer-compatible archive: {npz_path}")


if __name__ == "__main__":
    main()
