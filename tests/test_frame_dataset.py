import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import UTARLDDDataset
from src.frame_dataset import (
    build_npy_sequence_dataset,
    count_windows,
    discover_frame_feature_shards,
    iter_videos,
    save_frame_feature_shard,
)


class FrameDatasetTests(unittest.TestCase):
    def test_window_count(self):
        self.assertEqual(count_windows(10, seq_length=4, step_size=3), 3)
        self.assertEqual(count_windows(3, seq_length=4, step_size=1), 0)

    def test_fold_shards_rebuild_windows_without_crossing_video_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "features": np.full((5, 5), 1.0, dtype=np.float32),
                    "label": 0,
                    "subject_id": 1,
                    "fold_id": 1,
                    "video_path": "/dataset/01/0.mp4",
                },
                {
                    "features": np.full((5, 5), 2.0, dtype=np.float32),
                    "label": 1,
                    "subject_id": 2,
                    "fold_id": 1,
                    "video_path": "/dataset/02/5.mp4",
                },
            ]
            shard = root / "fold_1" / "frame_features_fold_1.npz"
            save_frame_feature_shard(shard, records, frame_skip=5)

            discovered = discover_frame_feature_shards([root])
            videos = list(iter_videos(discovered))
            paths, count = build_npy_sequence_dataset(
                videos, root / "sequences", 4, 1, False
            )

            self.assertEqual(count, 4)
            X = np.load(paths["X"])
            self.assertTrue(np.all(X[:2] == 1.0))
            self.assertTrue(np.all(X[2:] == 2.0))
            self.assertFalse(np.any((X.min(axis=(1, 2)) == 1) & (X.max(axis=(1, 2)) == 2)))

    def test_four_npy_directory_is_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(7)
            np.save(root / "X.npy", rng.normal(size=(9, 4, 5)).astype(np.float32))
            np.save(root / "Y.npy", np.tile(np.arange(3), 3).astype(np.int32))
            np.save(root / "subject.npy", np.repeat(np.arange(1, 4), 3).astype(np.int32))
            np.save(root / "folds.npy", np.repeat(np.arange(1, 4), 3).astype(np.int32))

            dataset = UTARLDDDataset.load_from_files(root)
            self.assertEqual(dataset.X.shape, (9, 4, 5))
            self.assertEqual(dataset.available_folds, [1, 2, 3])
            dataset.close()


if __name__ == "__main__":
    unittest.main()
