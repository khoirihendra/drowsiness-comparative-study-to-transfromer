"""
Dataset management, UTA-RLDD 5-fold cross-validation parsing,
and zero-leakage sliding-window temporal sequence generation.
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from config import (
    FOLD_SUBJECT_MAPPING,
    LABEL_MAP,
    MISSING_FEATURE_VECTOR,
    NUM_CLASSES,
    SEQ_LENGTH,
    STEP_SIZE,
    FEATURE_SUBSETS,
    TOTAL_RAW_FEATURES,
)


def to_categorical(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Convert integer class array into one-hot encoded matrix using pure NumPy."""
    y = np.asarray(y, dtype=np.int32)
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot



def parse_video_metadata(video_path: Union[str, Path]) -> Optional[Dict[str, Union[int, str]]]:
    """
    Parse label, subject ID, and fold ID from a UTA-RLDD video path.

    Supports standard UTA-RLDD folder structures:
    - Fold1/01/0.mp4, Fold2/15/5.mp4
    - 01/0.mp4, 15/5.mp4, 45/10.mov
    - 0_glasses.mp4, 5.mp4, 10.avi

    Returns:
        Dict with keys: 'label', 'subject_id', 'fold_id', 'video_path', 'filename'
        or None if label cannot be determined.
    """
    path = Path(video_path)
    filename = path.name.lower()
    stem = path.stem.lower()
    path_str = str(path).replace("\\", "/")

    # 1. Determine Class Label (0: Alert, 1: Low Vigilant, 2: Drowsy)
    label = None
    numeric_label = re.match(r"^(10|5|0)(?=$|[._\-\s(])", stem)
    if numeric_label:
        label = LABEL_MAP[numeric_label.group(1)]
    elif "alert" in stem:
        label = 0
    elif "low" in stem or "vigilant" in stem:
        label = 1
    elif "drowsy" in stem:
        label = 2

    if label is None:
        return None

    # 2. Determine Subject ID (1 to 60)
    subject_id = None
    # Search directory parts for subject folder (e.g. /1/, /01/, /subject_05/)
    for part in reversed(path.parts[:-1]):
        if re.search(r"fold", part, re.IGNORECASE):
            continue
        match = re.search(r"(\d+)", part)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 60:
                subject_id = num
                break

    # If subject_id is not represented in the path, do not silently merge the
    # video into subject 1: that would invalidate subject-independent splits.
    if subject_id is None:
        match = re.search(r"sub(?:ject)?[_-]?(\d+)", filename)
        if match:
            subject_id = int(match.group(1))
        else:
            return None

    # 3. Determine Fold ID (1 to 5)
    fold_id = None
    # Check if 'fold1' .. 'fold5' is explicitly in the path
    fold_match = re.search(r"fold[_-]?([1-5])", path_str, re.IGNORECASE)
    if fold_match:
        fold_id = int(fold_match.group(1))
    else:
        # Map subject_id to fold according to UTA-RLDD benchmark specification
        for f_idx, subjects in FOLD_SUBJECT_MAPPING.items():
            if subject_id in subjects:
                fold_id = f_idx
                break
        if fold_id is None:
            fold_id = ((subject_id - 1) // 12) + 1
            fold_id = min(max(fold_id, 1), 5)

    return {
        "label": label,
        "subject_id": subject_id,
        "fold_id": fold_id,
        "video_path": str(path),
        "filename": filename
    }


def find_all_video_files(dataset_dir: Union[str, Path]) -> List[str]:
    """Find all supported video files recursively."""
    valid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    dataset_path = Path(dataset_dir)
    video_files = []
    for ext in valid_exts:
        video_files.extend(glob.glob(str(dataset_path / "**" / f"*{ext}"), recursive=True))
        video_files.extend(glob.glob(str(dataset_path / "**" / f"*{ext.upper()}"), recursive=True))
    return sorted(list(set(video_files)))


def create_sliding_windows_for_video(
    video_features: np.ndarray,
    label: int,
    subject_id: int,
    fold_id: int,
    seq_length: int = SEQ_LENGTH,
    step_size: int = STEP_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate fixed-length temporal sequence windows for a single video.

    Prevents any cross-video temporal boundary leakage by operating strictly
    within each individual video's feature stream.

    Args:
        video_features: Array of shape (num_frames, num_features).
        label: Integer class label.
        subject_id: Integer subject ID.
        fold_id: Integer fold ID (1 to 5).
        seq_length: Number of timesteps per window (default: 30).
        step_size: Step stride between consecutive windows.

    Returns:
        X_windows: Array of shape (num_windows, seq_length, num_features)
        y_windows: Array of shape (num_windows,)
        subjects: Array of shape (num_windows,)
        folds: Array of shape (num_windows,)
    """
    num_frames = len(video_features)
    if num_frames < seq_length:
        return (
            np.empty((0, seq_length, video_features.shape[-1]), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    num_windows = (num_frames - seq_length) // step_size + 1
    X_windows = np.empty((num_windows, seq_length, video_features.shape[-1]), dtype=np.float32)
    y_windows = np.full((num_windows,), label, dtype=np.int32)
    sub_windows = np.full((num_windows,), subject_id, dtype=np.int32)
    fold_windows = np.full((num_windows,), fold_id, dtype=np.int32)

    for i in range(num_windows):
        start_idx = i * step_size
        X_windows[i] = video_features[start_idx : start_idx + seq_length]

    return X_windows, y_windows, sub_windows, fold_windows


class UTARLDDDataset:
    """
    Dataset container for UTA-RLDD features, supporting Zero-Leakage
    5-Fold Cross-Validation splits and feature subset slicing.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: np.ndarray,
        subjects: np.ndarray
    ):
        """
        Args:
            X: Array of shape (num_samples, seq_length, 5).
            y: Array of shape (num_samples,) containing class integers [0, 1, 2].
            folds: Array of shape (num_samples,) containing fold indices [1..5].
            subjects: Array of shape (num_samples,) containing subject IDs [1..60].
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.folds = folds.astype(np.int32)
        self.subjects = subjects.astype(np.int32)

    def __len__(self):
        return len(self.y)

    @property
    def available_folds(self) -> List[int]:
        """Return the sorted, non-empty fold IDs present in this archive."""
        return sorted(int(fold) for fold in np.unique(self.folds))

    def validate(self, chunk_size: int = 1_000_000) -> Dict[str, object]:
        """Validate shapes, metadata, leakage boundaries, and feature signal.

        The feature scan is chunked to avoid allocating another array as large as
        the extracted dataset.  A completely constant feature tensor is rejected,
        because it indicates a failed extraction rather than learnable input.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if self.X.ndim != 3:
            errors.append(f"X must be 3D (samples, timesteps, features), got {self.X.shape}.")
        elif self.X.shape[2] != TOTAL_RAW_FEATURES:
            errors.append(
                f"X must contain {TOTAL_RAW_FEATURES} raw features, got {self.X.shape[2]}."
            )

        sample_count = len(self.y)
        lengths = {
            "X": len(self.X),
            "y": len(self.y),
            "folds": len(self.folds),
            "subjects": len(self.subjects),
        }
        if len(set(lengths.values())) != 1:
            errors.append(f"Dataset arrays have inconsistent sample counts: {lengths}.")
        if sample_count == 0:
            errors.append("Dataset contains no samples.")

        invalid_labels = sorted(set(np.unique(self.y).tolist()) - set(range(NUM_CLASSES)))
        if invalid_labels:
            errors.append(f"Labels outside [0, {NUM_CLASSES - 1}]: {invalid_labels}.")

        available_folds = self.available_folds
        if len(available_folds) < 3:
            errors.append(
                "At least 3 non-empty folds are required for separate train/validation/test sets; "
                f"found {available_folds}."
            )

        if len(self.subjects) == len(self.folds):
            subject_to_folds = {
                int(subject): sorted(
                    int(fold) for fold in np.unique(self.folds[self.subjects == subject])
                )
                for subject in np.unique(self.subjects)
            }
            leaked_subjects = {
                subject: folds for subject, folds in subject_to_folds.items() if len(folds) != 1
            }
            if leaked_subjects:
                errors.append(f"Subjects assigned to multiple folds: {leaked_subjects}.")

        invalid_folds = sorted(
            int(fold) for fold in np.unique(self.folds) if int(fold) not in range(1, 6)
        )
        if invalid_folds:
            errors.append(f"Fold IDs must be in [1, 5], got {invalid_folds}.")
        if np.any(self.subjects < 1):
            errors.append("Subject IDs must be positive integers.")

        padding_count = 0
        total_timesteps = 0
        feature_min = None
        feature_max = None
        if self.X.ndim == 3 and self.X.shape[2] > 0 and len(self.X) > 0:
            flat = self.X.reshape(-1, self.X.shape[2])
            feature_min = np.full(self.X.shape[2], np.inf, dtype=np.float64)
            feature_max = np.full(self.X.shape[2], -np.inf, dtype=np.float64)
            padding = np.asarray(MISSING_FEATURE_VECTOR, dtype=self.X.dtype)

            for start in range(0, len(flat), chunk_size):
                chunk = flat[start : start + chunk_size]
                if not np.isfinite(chunk).all():
                    errors.append("Feature tensor contains NaN or infinite values.")
                    break
                feature_min = np.minimum(feature_min, np.min(chunk, axis=0))
                feature_max = np.maximum(feature_max, np.max(chunk, axis=0))
                if len(padding) == chunk.shape[1]:
                    padding_count += int(np.count_nonzero(np.all(chunk == padding, axis=1)))
                total_timesteps += len(chunk)

            if np.all(feature_min == feature_max):
                errors.append(
                    "All extracted features are constant. This usually means MediaPipe never "
                    "reached the detector and every frame was replaced by padding. Re-extract "
                    "the raw videos; this NPZ cannot be used for training."
                )
            else:
                constant_indices = np.flatnonzero(feature_min == feature_max).tolist()
                if constant_indices:
                    warnings.append(f"Constant feature columns detected: {constant_indices}.")

        padding_rate = padding_count / total_timesteps if total_timesteps else 0.0
        if padding_rate >= 0.99:
            errors.append(f"Padding rate is {padding_rate:.2%}; extraction produced no usable signal.")
        elif padding_rate > 0.20:
            warnings.append(f"High missing-face padding rate: {padding_rate:.2%}.")

        class_counts = [int(np.count_nonzero(self.y == label)) for label in range(NUM_CLASSES)]
        fold_counts = {
            fold: int(np.count_nonzero(self.folds == fold)) for fold in available_folds
        }
        summary: Dict[str, object] = {
            "num_samples": sample_count,
            "num_subjects": int(len(np.unique(self.subjects))),
            "available_folds": available_folds,
            "class_counts": class_counts,
            "fold_counts": fold_counts,
            "padding_rate": float(padding_rate),
            "feature_min": feature_min.tolist() if feature_min is not None else [],
            "feature_max": feature_max.tolist() if feature_max is not None else [],
            "warnings": warnings,
        }

        if errors:
            raise ValueError("Dataset integrity check failed:\n- " + "\n- ".join(errors))
        return summary

    @classmethod
    def load_from_files(cls, npz_or_npy_path: Union[str, Path]) -> "UTARLDDDataset":
        """
        Load dataset from saved .npz archive or legacy .npy files.
        """
        path = Path(npz_or_npy_path)
        if path.suffix == ".npz":
            data = np.load(path)
            return cls(
                X=data["X"],
                y=data["y"],
                folds=data["folds"],
                subjects=data["subjects"]
            )
        elif path.suffix == ".npy":
            raise ValueError(
                "Legacy .npy feature files do not contain subject/fold metadata and cannot "
                "support leakage-safe evaluation. Re-run extract_features.py to create an .npz "
                "archive with X, y, folds, and subjects."
            )
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def save_npz(self, output_path: Union[str, Path]) -> None:
        """Save dataset to compressed .npz archive."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            X=self.X,
            y=self.y,
            folds=self.folds,
            subjects=self.subjects
        )

    def get_fold_split(
        self,
        test_fold: int,
        val_fold: Optional[int] = None,
        feature_subset: str = "all",
        one_hot: bool = True
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Generate Train, Validation, and Test splits according to UTA-RLDD
        5-fold cross-validation protocol with STRICT ZERO SUBJECT LEAKAGE.

        Args:
            test_fold: Fold index (1 to 5) to use as the held-out test set.
            val_fold: Fold index for validation. If None, automatically select
                      the next circular fold (e.g. test_fold=1 -> val_fold=2).
            feature_subset: 'ear' (1 feat), 'ear_mar' (2 feats), or 'all' (5 feats).
            one_hot: If True, convert y to one-hot categorical vectors (N, 3).

        Returns:
            (X_train, y_train, y_train_raw),
            (X_val, y_val, y_val_raw),
            (X_test, y_test, y_test_raw)
        """
        if test_fold < 1 or test_fold > 5:
            raise ValueError(f"test_fold must be between 1 and 5, got {test_fold}")

        available_folds = self.available_folds
        if test_fold not in available_folds:
            raise ValueError(
                f"test_fold {test_fold} is empty or absent; available folds: {available_folds}"
            )
        if len(available_folds) < 3:
            raise ValueError(
                f"Need at least 3 folds for train/validation/test, found {available_folds}."
            )

        if val_fold is None:
            # Deterministic validation fold from the folds actually present.
            test_position = available_folds.index(test_fold)
            val_fold = available_folds[(test_position + 1) % len(available_folds)]

        if val_fold not in available_folds:
            raise ValueError(
                f"val_fold {val_fold} is empty or absent; available folds: {available_folds}"
            )

        if val_fold == test_fold:
            raise ValueError(f"val_fold ({val_fold}) cannot be the same as test_fold ({test_fold})")

        # Slice feature columns
        feat_info = FEATURE_SUBSETS.get(feature_subset, FEATURE_SUBSETS["all"])
        indices = feat_info["indices"]
        X_sliced = self.X[:, :, indices]

        # Partition masks (100% subject-disjoint)
        test_mask = (self.folds == test_fold)
        val_mask = (self.folds == val_fold)
        train_mask = (~test_mask) & (~val_mask)

        X_train, y_train_raw = X_sliced[train_mask], self.y[train_mask]
        X_val, y_val_raw = X_sliced[val_mask], self.y[val_mask]
        X_test, y_test_raw = X_sliced[test_mask], self.y[test_mask]

        split_labels = {
            "train": np.unique(y_train_raw).tolist(),
            "validation": np.unique(y_val_raw).tolist(),
            "test": np.unique(y_test_raw).tolist(),
        }
        expected_labels = list(range(NUM_CLASSES))
        invalid_splits = {
            name: labels for name, labels in split_labels.items() if labels != expected_labels
        }
        if invalid_splits:
            raise ValueError(
                f"Every split must contain labels {expected_labels}; got {invalid_splits}."
            )

        if one_hot:
            y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
            y_val = to_categorical(y_val_raw, num_classes=NUM_CLASSES)
            y_test = to_categorical(y_test_raw, num_classes=NUM_CLASSES)
        else:
            y_train = y_train_raw
            y_val = y_val_raw
            y_test = y_test_raw

        return (
            (X_train, y_train, y_train_raw),
            (X_val, y_val, y_val_raw),
            (X_test, y_test, y_test_raw)
        )
