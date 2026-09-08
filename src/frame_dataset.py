"""Frame-level feature shards and configurable temporal-window generation.

The expensive operation in this project is MediaPipe inference.  This module
stores its output per fold while retaining video boundaries, so temporal
windows can be rebuilt later without reading the source videos again.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from numpy.lib.format import open_memmap

from config import MISSING_FEATURE_VECTOR, NUM_CLASSES, TOTAL_RAW_FEATURES


RAW_FORMAT_VERSION = 1
RAW_SHARD_PATTERN = "frame_features_fold_*.npz"


def save_frame_feature_shard(
    output_path: Union[str, Path],
    records: Sequence[Dict[str, object]],
    frame_skip: int,
) -> None:
    """Save frame features for multiple videos without losing boundaries."""
    if not records:
        raise ValueError("Cannot save an empty frame-feature shard.")

    feature_arrays = [np.asarray(record["features"], dtype=np.float32) for record in records]
    for features in feature_arrays:
        if features.ndim != 2 or features.shape[1] != TOTAL_RAW_FEATURES:
            raise ValueError(
                f"Each video must have shape (frames, {TOTAL_RAW_FEATURES}); got {features.shape}."
            )

    lengths = np.asarray([len(features) for features in feature_arrays], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    features = np.concatenate(feature_arrays, axis=0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".partial")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            format_version=np.asarray(RAW_FORMAT_VERSION, dtype=np.int32),
            features=features,
            video_offsets=offsets,
            labels=np.asarray([record["label"] for record in records], dtype=np.int32),
            subjects=np.asarray([record["subject_id"] for record in records], dtype=np.int32),
            folds=np.asarray([record["fold_id"] for record in records], dtype=np.int32),
            video_paths=np.asarray([str(record["video_path"]) for record in records]),
            frame_skip=np.asarray(frame_skip, dtype=np.int32),
        )
    temporary_path.replace(path)


def load_frame_feature_shard(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load and validate a frame-feature fold archive."""
    shard_path = Path(path)
    with np.load(shard_path, allow_pickle=False) as archive:
        required = {
            "format_version",
            "features",
            "video_offsets",
            "labels",
            "subjects",
            "folds",
            "video_paths",
            "frame_skip",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{shard_path} is not a frame-feature shard; missing {sorted(missing)}.")
        data = {name: archive[name] for name in required}

    if int(data["format_version"]) != RAW_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported frame-feature format {int(data['format_version'])} in {shard_path}."
        )
    features = data["features"]
    offsets = data["video_offsets"]
    video_count = len(data["labels"])
    if features.ndim != 2 or features.shape[1] != TOTAL_RAW_FEATURES:
        raise ValueError(f"Invalid feature shape in {shard_path}: {features.shape}.")
    if len(offsets) != video_count + 1 or offsets[0] != 0 or offsets[-1] != len(features):
        raise ValueError(f"Invalid video offsets in {shard_path}.")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"Video offsets are not monotonic in {shard_path}.")
    for key in ("subjects", "folds", "video_paths"):
        if len(data[key]) != video_count:
            raise ValueError(f"Metadata length mismatch for {key} in {shard_path}.")
    if not np.isfinite(features).all():
        raise ValueError(f"NaN or infinite frame features found in {shard_path}.")
    invalid_labels = sorted(set(np.unique(data["labels"]).tolist()) - set(range(NUM_CLASSES)))
    if invalid_labels:
        raise ValueError(f"Invalid labels in {shard_path}: {invalid_labels}.")
    return data


def discover_frame_feature_shards(inputs: Iterable[Union[str, Path]]) -> List[Path]:
    """Resolve explicit archives and directories into unique fold shards."""
    found: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            found.extend(path.rglob(RAW_SHARD_PATTERN))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"Frame-feature input does not exist: {path}")

    unique = sorted({path.resolve() for path in found})
    if not unique:
        raise FileNotFoundError(
            f"No files matching {RAW_SHARD_PATTERN!r} were found in the supplied inputs."
        )
    return unique


def iter_videos(
    shard_paths: Iterable[Union[str, Path]],
) -> Iterable[Tuple[np.ndarray, int, int, int, str]]:
    """Yield (features, label, subject, fold, path) for every stored video."""
    seen_paths = set()
    for shard_path in shard_paths:
        shard = load_frame_feature_shard(shard_path)
        offsets = shard["video_offsets"]
        for index in range(len(shard["labels"])):
            video_path = str(shard["video_paths"][index])
            identity = (int(shard["subjects"][index]), int(shard["labels"][index]), video_path)
            if identity in seen_paths:
                raise ValueError(f"Duplicate video found across frame-feature shards: {video_path}")
            seen_paths.add(identity)
            start, end = int(offsets[index]), int(offsets[index + 1])
            yield (
                shard["features"][start:end],
                int(shard["labels"][index]),
                int(shard["subjects"][index]),
                int(shard["folds"][index]),
                video_path,
            )


def count_windows(video_length: int, seq_length: int, step_size: int) -> int:
    """Return how many complete windows fit in one video."""
    if seq_length <= 0:
        raise ValueError("seq_length must be greater than zero.")
    if step_size <= 0:
        raise ValueError("step_size must be greater than zero.")
    return max(0, (video_length - seq_length) // step_size + 1)


def build_npy_sequence_dataset(
    videos: Sequence[Tuple[np.ndarray, int, int, int, str]],
    output_dir: Union[str, Path],
    seq_length: int,
    step_size: int,
    overwrite: bool = False,
) -> Tuple[Dict[str, Path], int]:
    """Write four memory-efficient NPY arrays from frame-level video records."""
    destination = Path(output_dir)
    paths = {
        "X": destination / "X.npy",
        "y": destination / "Y.npy",
        "subjects": destination / "subject.npy",
        "folds": destination / "folds.npy",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already exists ({existing[0]}). Use --overwrite to replace all four arrays."
        )
    destination.mkdir(parents=True, exist_ok=True)
    total_windows = sum(
        count_windows(len(features), seq_length, step_size) for features, *_ in videos
    )
    if total_windows == 0:
        raise ValueError("No video is long enough for the requested sequence length.")

    partials = {key: path.with_name(path.name + ".partial.npy") for key, path in paths.items()}
    arrays = {
        "X": open_memmap(
            partials["X"],
            mode="w+",
            dtype=np.float32,
            shape=(total_windows, seq_length, TOTAL_RAW_FEATURES),
        ),
        "y": open_memmap(partials["y"], mode="w+", dtype=np.int32, shape=(total_windows,)),
        "subjects": open_memmap(
            partials["subjects"], mode="w+", dtype=np.int32, shape=(total_windows,)
        ),
        "folds": open_memmap(
            partials["folds"], mode="w+", dtype=np.int32, shape=(total_windows,)
        ),
    }

    cursor = 0
    for features, label, subject, fold, _ in videos:
        number = count_windows(len(features), seq_length, step_size)
        for local_index in range(number):
            start = local_index * step_size
            arrays["X"][cursor + local_index] = features[start : start + seq_length]
        end = cursor + number
        arrays["y"][cursor:end] = label
        arrays["subjects"][cursor:end] = subject
        arrays["folds"][cursor:end] = fold
        cursor = end

    for array in arrays.values():
        array.flush()
    del array
    del arrays
    for key, final_path in paths.items():
        os.replace(partials[key], final_path)
    return paths, total_windows


def frame_padding_rate(features: np.ndarray) -> float:
    """Compute the missing-face rate in a 2-D frame feature array."""
    if len(features) == 0:
        return 0.0
    padding = np.asarray(MISSING_FEATURE_VECTOR, dtype=np.float32)
    return float(np.mean(np.all(features == padding, axis=1)))
