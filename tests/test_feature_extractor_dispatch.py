import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


class FeatureExtractorDispatchTests(unittest.TestCase):
    def test_tasks_api_mode_reaches_detector(self):
        with patch.dict(sys.modules, {"cv2": MagicMock()}):
            module = importlib.import_module("src.feature_extractor")

        detector = MagicMock()
        detector.detect.return_value = SimpleNamespace(face_landmarks=[["landmarks"]])
        pipeline = module.FacialLandmarkerPipeline.__new__(module.FacialLandmarkerPipeline)
        pipeline.mode = "tasks_api"
        pipeline.detector = detector

        fake_mp = SimpleNamespace(
            Image=lambda **kwargs: kwargs,
            ImageFormat=SimpleNamespace(SRGB="srgb"),
        )
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        expected = (0.2, 0.4, 1.0, 2.0, 3.0)

        with patch.object(module, "mp", fake_mp, create=True), patch.object(
            module,
            "extract_facial_features_from_landmarks",
            return_value=expected,
        ):
            actual = pipeline.process_frame(frame, frame_w=2, frame_h=2)

        self.assertEqual(actual, expected)
        detector.detect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
