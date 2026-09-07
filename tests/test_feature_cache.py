import unittest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import MISSING_FEATURE_VECTOR
import build_windows
from src.feature_cache import save_cache_npz, window_cached_video


class FeatureCacheWindowTests(unittest.TestCase):
    def test_frame_skip_sequence_and_step_are_applied_after_cache(self):
        features = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
        observed = np.ones(12, dtype=np.bool_)
        frames = np.arange(12, dtype=np.int64)
        timestamps = frames / 10.0

        result = window_cached_video(
            features,
            observed,
            frames,
            timestamps,
            frame_skip=2,
            sequence_length=3,
            step_size=2,
        )

        self.assertEqual(result["X"].shape, (2, 3, 5))
        np.testing.assert_array_equal(result["X"][0], features[[0, 2, 4]])
        np.testing.assert_array_equal(result["X"][1], features[[4, 6, 8]])
        np.testing.assert_array_equal(result["window_start_frame"], [0, 4])
        np.testing.assert_array_equal(result["window_end_frame"], [4, 8])

    def test_missing_detection_is_padded_but_mask_is_preserved(self):
        features = np.ones((5, 5), dtype=np.float32)
        features[2] = np.nan
        observed = np.array([True, True, False, True, True])
        frames = np.arange(5, dtype=np.int64)

        result = window_cached_video(
            features,
            observed,
            frames,
            frames.astype(np.float64),
            frame_skip=1,
            sequence_length=5,
            step_size=1,
        )

        np.testing.assert_array_equal(result["observed_mask"][0], observed)
        np.testing.assert_allclose(result["X"][0, 2], MISSING_FEATURE_VECTOR)
        self.assertTrue(np.isfinite(result["X"]).all())

    def test_windows_never_cross_video_boundary(self):
        arrays = []
        for value in (1.0, 9.0):
            features = np.full((4, 5), value, dtype=np.float32)
            arrays.append(
                window_cached_video(
                    features,
                    np.ones(4, dtype=np.bool_),
                    np.arange(4),
                    np.arange(4, dtype=np.float64),
                    frame_skip=1,
                    sequence_length=3,
                    step_size=1,
                )["X"]
            )

        combined = np.concatenate(arrays)
        self.assertTrue(all(np.unique(window).size == 1 for window in combined))

    def test_builder_output_is_training_compatible_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos_dir = root / "videos"
            videos_dir.mkdir()
            entries = []
            frame_indices = np.arange(6, dtype=np.int64)

            for fold in range(1, 4):
                for label in range(3):
                    video_id = f"f{fold}_c{label}"
                    cache_file = videos_dir / f"{video_id}.npz"
                    features = np.arange(30, dtype=np.float32).reshape(6, 5) + fold + label
                    metadata = {
                        "video_id": video_id,
                        "subject_id": fold,
                        "fold_id": fold,
                        "label": label,
                    }
                    save_cache_npz(
                        cache_file,
                        {
                            "features": features,
                            "observed_mask": np.ones(6, dtype=np.bool_),
                            "frame_indices": frame_indices,
                            "timestamps_sec": frame_indices / 10.0,
                        },
                        metadata,
                    )
                    entries.append({**metadata, "cache_file": str(Path("videos") / cache_file.name)})

            (root / "manifest.json").write_text(
                json.dumps({"format_version": 1, "status": "complete", "videos": entries}),
                encoding="utf-8",
            )
            output = root / "windows.npz"
            argv = [
                "build_windows.py",
                "--cache_dir", str(root),
                "--output_path", str(output),
                "--frame_skip", "1",
                "--sequence_length", "3",
                "--step_size", "2",
            ]
            with patch.object(sys, "argv", argv):
                build_windows.main()

            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["X"].shape, (18, 3, 5))
                self.assertEqual(set(archive.files), {
                    "X", "y", "folds", "subjects", "observed_mask", "video_ids",
                    "video_keys", "window_start_frame", "window_end_frame",
                    "window_start_sec", "window_end_sec", "build_metadata_json",
                })
                for video_index in np.unique(archive["video_ids"]):
                    indices = archive["video_ids"] == video_index
                    self.assertEqual(len(np.unique(archive["subjects"][indices])), 1)
                    self.assertEqual(len(np.unique(archive["folds"][indices])), 1)


if __name__ == "__main__":
    unittest.main()
