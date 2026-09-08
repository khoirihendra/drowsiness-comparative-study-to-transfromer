"""Extract UTA-RLDD frame features without constructing temporal windows."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import (
    DEFAULT_DATASET_PATH, DEFAULT_LANDMARKER_MODEL_PATH, FEATURES_DIR,
    FEATURE_NAMES, FRAME_RESIZE, MAX_ALLOWED_PADDING_RATE, MISSING_FEATURE_VECTOR,
)
from src.dataset import find_all_video_files, parse_video_metadata


def extract_raw_dataset(dataset_paths, output_path, model_path=None, use_gpu=False,
                        frame_skip=1, max_frames=None,
                        max_padding_rate=MAX_ALLOWED_PADDING_RATE):
    """Save aligned numeric frame arrays and a manifest of [start, stop) video slices.

    All decoded sampled frames are retained, including missing-face placeholders.
    No normalization, temporal padding, truncation to windows, or windowing is done.
    Temporary per-video arrays keep memory usage bounded by one video.
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be positive.")
    if not 0 <= max_padding_rate <= 1:
        raise ValueError("max_padding_rate must be between 0 and 1.")
    output = Path(output_path)
    names = ("x.npy", "y.npy", "subjects.npy", "folds.npy", "manifest.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError(f"Extraction files already exist in {output}; choose a new directory.")
    videos = sorted({str(Path(video).resolve()) for root in dataset_paths
                     for video in find_all_video_files(root)})
    if not videos:
        raise ValueError("No supported videos found in dataset_paths.")

    # Validate metadata before starting expensive inference. Reuse existing label/fold rules.
    metadata = []
    subject_folds = {}
    for video in videos:
        meta = parse_video_metadata(video)
        if meta is None:
            raise ValueError(f"Cannot determine label/subject/fold for {video}")
        subject, fold = meta["subject_id"], meta["fold_id"]
        if not 1 <= subject <= 60 or not 1 <= fold <= 5:
            raise ValueError(f"Invalid subject/fold metadata for {video}: {meta}")
        if subject in subject_folds and subject_folds[subject] != fold:
            raise ValueError(f"Subject {subject} is assigned to multiple folds.")
        subject_folds[subject] = fold
        metadata.append(meta)

    from src.feature_extractor import FacialLandmarkerPipeline, extract_features_from_video

    output.mkdir(parents=True, exist_ok=True)
    records = []
    total = padding_count = 0
    feature_min = np.full(5, np.inf)
    feature_max = np.full(5, -np.inf)
    with TemporaryDirectory(prefix="raw-extraction-", dir=output) as staging:
        staging = Path(staging)
        for index, meta in enumerate(metadata):
            print(f"[{index + 1}/{len(metadata)}] {meta['video_path']}", flush=True)
            # Reset tracking between recordings, including with the FaceMesh fallback.
            pipeline = FacialLandmarkerPipeline(model_asset_path=model_path, use_gpu=use_gpu)
            try:
                features = extract_features_from_video(
                    meta["video_path"], pipeline, frame_skip=frame_skip,
                    max_frames=max_frames, resize_dim=FRAME_RESIZE,
                )
            finally:
                pipeline.close()
            features = np.asarray(features, dtype=np.float32)
            if features.ndim != 2 or features.shape[1] != 5 or not len(features):
                raise ValueError(f"No valid frame features extracted from {meta['video_path']}")
            if not np.isfinite(features).all():
                raise ValueError(f"Non-finite features in {meta['video_path']}")
            missing = int(np.count_nonzero(np.all(
                features == np.asarray(MISSING_FEATURE_VECTOR, dtype=np.float32), axis=1)))
            padding_count += missing
            feature_min = np.minimum(feature_min, features.min(axis=0))
            feature_max = np.maximum(feature_max, features.max(axis=0))
            np.save(staging / f"video_{index}.npy", features, allow_pickle=False)
            records.append({**meta, "start": total, "stop": total + len(features),
                            "num_frames": len(features), "padding_frames": missing})
            total += len(features)

        padding_rate = padding_count / total
        if padding_rate > max_padding_rate or padding_rate >= 0.99:
            raise ValueError(f"Missing-face padding rate {padding_rate:.2%} exceeds quality limits.")
        if np.all(feature_min == feature_max):
            raise ValueError("All extracted features are constant; check the face detector.")

        # Assemble one contiguous array per field, without loading all videos into RAM.
        fields = {"x": (np.float32, (total, 5)), "y": (np.int32, (total,)),
                  "subjects": (np.int32, (total,)), "folds": (np.int32, (total,))}
        for field, (dtype, shape) in fields.items():
            array = np.lib.format.open_memmap(staging / f"{field}.npy", mode="w+",
                                              dtype=dtype, shape=shape)
            try:
                for index, record in enumerate(records):
                    start, stop = record["start"], record["stop"]
                    if field == "x":
                        array[start:stop] = np.load(staging / f"video_{index}.npy", allow_pickle=False)
                    else:
                        key = {"y": "label", "subjects": "subject_id", "folds": "fold_id"}[field]
                        array[start:stop] = record[key]
                array.flush()
            finally:
                del array

        manifest = {
            "format_version": 1, "layout": "frames", "feature_names": FEATURE_NAMES,
            "num_frames": total, "frame_skip": frame_skip,
            "max_frames_per_video": max_frames if max_frames and max_frames > 0 else None,
            "resize_dim": list(FRAME_RESIZE), "missing_feature_vector": list(MISSING_FEATURE_VECTOR),
            "padding_rate": padding_rate,
            "label_names": {"0": "Alert", "1": "Low Vigilant", "2": "Drowsy"},
            "frame_index_rule": "Within each video, row i samples source frame i * frame_skip (zero-based).",
            "slice_rule": "Each video occupies x[start:stop]; stop is exclusive. Window each slice separately.",
            "videos": records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for name in names:
            (staging / name).replace(output / name)
    print(f"Saved {total} frames from {len(records)} videos to {output}; x.shape=({total}, 5)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", nargs="+", default=[DEFAULT_DATASET_PATH])
    parser.add_argument("--output_path", default=str(FEATURES_DIR / "uta_rldd_raw"),
                        help="Output directory for x.npy, y.npy, subjects.npy, folds.npy and manifest.json.")
    parser.add_argument("--model_path", default=DEFAULT_LANDMARKER_MODEL_PATH)
    parser.add_argument("--frame_skip", type=int, default=1, help="Sample every Nth frame (default: every frame).")
    parser.add_argument("--max_frames", type=int, default=None, help="Sampled frames per video; omitted or <=0 means all.")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--max_padding_rate", type=float, default=MAX_ALLOWED_PADDING_RATE)
    args = parser.parse_args()
    extract_raw_dataset(args.dataset_path, args.output_path, args.model_path, args.use_gpu,
                        args.frame_skip, args.max_frames, args.max_padding_rate)


if __name__ == "__main__":
    main()
