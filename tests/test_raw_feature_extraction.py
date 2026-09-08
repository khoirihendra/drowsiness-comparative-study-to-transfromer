import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from extract_raw_features import extract_raw_dataset
from src.dataset import create_sliding_windows_for_video


class RawFeatureExtractionTests(unittest.TestCase):
    def run_extraction(self, root, features, **kwargs):
        videos = [str(Path(root) / "01" / name) for name in ("0.mp4", "0_glasses.mp4")]
        pipeline = MagicMock()
        extractor = MagicMock(side_effect=features)
        backend = SimpleNamespace(FacialLandmarkerPipeline=MagicMock(return_value=pipeline),
                                  extract_features_from_video=extractor)
        with patch("extract_raw_features.find_all_video_files", return_value=videos), patch.dict(
            "sys.modules", {"src.feature_extractor": backend}
        ):
            manifest = extract_raw_dataset([root], Path(root) / "out", **kwargs)
        self.assertEqual(pipeline.close.call_count, 2)
        return manifest, extractor

    def test_round_trip_keeps_short_videos_and_separate_recordings(self):
        a = np.arange(15, dtype=np.float32).reshape(3, 5)
        b = np.arange(20, 40, dtype=np.float32).reshape(4, 5)
        with tempfile.TemporaryDirectory() as root:
            manifest, extractor = self.run_extraction(root, [a, b], frame_skip=2)
            output = Path(root) / "out"
            x = np.load(output / "x.npy", allow_pickle=False, mmap_mode="r")
            np.testing.assert_array_equal(x, np.concatenate([a, b]))
            self.assertEqual(x.dtype, np.float32)
            for name, value in (("y", 0), ("subjects", 1), ("folds", 1)):
                array = np.load(output / f"{name}.npy", allow_pickle=False)
                np.testing.assert_array_equal(array, np.full(7, value))
                self.assertEqual(array.dtype, np.int32)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
            windows = [create_sliding_windows_for_video(
                x[v["start"]:v["stop"]], v["label"], v["subject_id"], v["fold_id"],
                seq_length=3, step_size=1)[0] for v in manifest["videos"]]
            self.assertEqual([len(w) for w in windows], [1, 2])
            np.testing.assert_array_equal(windows[1][0], b[:3])
            self.assertEqual(extractor.call_args.kwargs["frame_skip"], 2)
            del x
            with self.assertRaises(FileExistsError):
                extract_raw_dataset([root], output)

    def test_broken_extraction_does_not_publish_arrays(self):
        from config import MISSING_FEATURE_VECTOR
        missing = np.tile(MISSING_FEATURE_VECTOR, (3, 1))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "padding rate"):
                self.run_extraction(root, [missing, missing])
            self.assertEqual(list((Path(root) / "out").iterdir()), [])

    def test_invalid_sampling_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame_skip"):
            extract_raw_dataset([], "unused", frame_skip=0)


if __name__ == "__main__":
    unittest.main()
