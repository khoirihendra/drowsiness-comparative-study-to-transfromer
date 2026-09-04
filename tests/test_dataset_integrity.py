import unittest

import numpy as np

from config import MISSING_FEATURE_VECTOR
from src.dataset import UTARLDDDataset, parse_video_metadata


def make_dataset(num_folds=4):
    samples_per_fold = 6
    num_samples = num_folds * samples_per_fold
    rng = np.random.default_rng(42)
    X = rng.normal(size=(num_samples, 30, 5)).astype(np.float32)
    y = np.tile(np.arange(3, dtype=np.int32), num_samples // 3)
    folds = np.repeat(np.arange(1, num_folds + 1, dtype=np.int32), samples_per_fold)
    subjects = np.repeat(np.arange(1, num_folds + 1, dtype=np.int32), samples_per_fold)
    return UTARLDDDataset(X=X, y=y, folds=folds, subjects=subjects)


class DatasetIntegrityTests(unittest.TestCase):
    def test_metadata_parser_distinguishes_ten_from_zero(self):
        metadata = parse_video_metadata("/dataset/Fold2_part1/13/10_glasses.mp4")

        self.assertEqual(metadata["label"], 2)
        self.assertEqual(metadata["subject_id"], 13)
        self.assertEqual(metadata["fold_id"], 2)

    def test_metadata_parser_does_not_treat_fold_number_as_subject(self):
        self.assertIsNone(parse_video_metadata("/dataset/Fold2_part1/10.mp4"))

    def test_rejects_all_padding_archive(self):
        X = np.tile(
            np.asarray(MISSING_FEATURE_VECTOR, dtype=np.float32),
            (12, 30, 1),
        )
        dataset = UTARLDDDataset(
            X=X,
            y=np.tile(np.arange(3, dtype=np.int32), 4),
            folds=np.repeat(np.arange(1, 4, dtype=np.int32), 4),
            subjects=np.repeat(np.arange(1, 4, dtype=np.int32), 4),
        )

        with self.assertRaisesRegex(ValueError, "constant"):
            dataset.validate(chunk_size=10)

    def test_four_fold_validation_wraps_to_first_available_fold(self):
        dataset = make_dataset(num_folds=4)
        (_, _, _), (X_val, _, _), (X_test, _, _) = dataset.get_fold_split(test_fold=4)

        np.testing.assert_array_equal(X_val, dataset.X[dataset.folds == 1])
        np.testing.assert_array_equal(X_test, dataset.X[dataset.folds == 4])

    def test_rejects_subject_assigned_to_multiple_folds(self):
        dataset = make_dataset(num_folds=4)
        dataset.subjects[6] = dataset.subjects[0]

        with self.assertRaisesRegex(ValueError, "multiple folds"):
            dataset.validate()

    def test_rejects_legacy_npy_without_group_metadata(self):
        with self.assertRaisesRegex(ValueError, "cannot support leakage-safe evaluation"):
            UTARLDDDataset.load_from_files("legacy.npy")


if __name__ == "__main__":
    unittest.main()
