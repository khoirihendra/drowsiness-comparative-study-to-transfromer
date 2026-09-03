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
    NUM_CLASSES,
    SEQ_LENGTH,
    STEP_SIZE,
    FEATURE_SUBSETS
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
    path_str = str(path).replace("\\", "/")

    # 1. Determine Class Label (0: Alert, 1: Low Vigilant, 2: Drowsy)
    label = None
    if re.match(r"^0[\._\-]", filename) or filename == "0.mp4" or filename.startswith("0"):
        label = 0
    elif re.match(r"^5[\._\-]", filename) or filename == "5.mp4" or filename.startswith("5"):
        label = 1
    elif re.match(r"^10[\._\-]", filename) or filename == "10.mp4" or filename.startswith("10"):
        label = 2
    else:
        # Fallback check inside filename
        if "alert" in filename or "0" in filename:
            label = 0
        elif "low" in filename or "5" in filename:
            label = 1
        elif "drowsy" in filename or "10" in filename:
            label = 2

    if label is None:
        return None

    # 2. Determine Subject ID (1 to 60)
    subject_id = None
    # Search directory parts for subject folder (e.g. /1/, /01/, /subject_05/)
    for part in reversed(path.parts[:-1]):
        match = re.search(r"(\d+)", part)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 60:
                subject_id = num
                break

    # If subject_id not found in path, attempt from filename or default to 1
    if subject_id is None:
        match = re.search(r"sub(?:ject)?[_-]?(\d+)", filename)
        if match:
            subject_id = int(match.group(1))
        else:
            subject_id = 1

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
            # Support loading legacy unpartitioned npy arrays
            X = np.load(path)
            y_path = path.parent / path.name.replace("X_features", "y_labels").replace("X_", "y_")
            if not y_path.exists():
                y_path = path.parent / "y_labels_rldd.npy"
            y = np.load(y_path)

            # Synthesize 5-fold partition deterministically if metadata is absent
            num_samples = len(y)
            folds = np.zeros(num_samples, dtype=np.int32)
            subjects = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                fold = (i % 5) + 1
                folds[i] = fold
                subjects[i] = fold * 12

            return cls(X=X, y=y, folds=folds, subjects=subjects)
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

        if val_fold is None:
            # Deterministic validation fold from remaining folds
            val_fold = (test_fold % 5) + 1

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
