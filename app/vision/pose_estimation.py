"""
Stage 3 - BRANCH 1: POSE REGRESSION NODE
Wraps a YOLOv8-pose model trained on your keypoints dataset
(posture-types-image-dataset-frontback-view / posture-keypoints notebook).

Expected keypoint order the model should be trained/exported with:
    0: C7 vertebra (base of neck)
    1: left shoulder
    2: right shoulder
    3: curve peak (apex of lateral deviation, coronal view)
    4: left pelvic crest
    5: right pelvic crest
    6: sacral base (mid-lumbosacral point)

If no trained weights are present yet, this falls back to a deterministic
mock so the rest of the pipeline (angle math, tagging, reporting) is fully
testable before training finishes. Swap `MOCK_MODE` off once you drop your
.pt file into /weights.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.config import settings

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


@dataclass
class PoseKeypoints:
    c7: tuple[float, float]
    left_shoulder: tuple[float, float]
    right_shoulder: tuple[float, float]
    curve_peak: tuple[float, float]
    left_pelvis: tuple[float, float]
    right_pelvis: tuple[float, float]
    sacral_base: tuple[float, float]
    confidence: float


class PoseEstimator:
    def __init__(self, weights_filename: str = settings.pose_model_weights):
        self.weights_path = settings.weights_dir / weights_filename
        self.model = None
        self.mock_mode = True

        if _ULTRALYTICS_AVAILABLE and self.weights_path.exists():
            self.model = YOLO(str(self.weights_path))
            self.mock_mode = False

    def predict(self, image_bgr: np.ndarray) -> PoseKeypoints:
        if self.mock_mode:
            return self._mock_predict(image_bgr)
        return self._model_predict(image_bgr)

    def _model_predict(self, image_bgr: np.ndarray) -> PoseKeypoints:
        results = self.model.predict(source=image_bgr, verbose=False)
        r = results[0]
        kp = r.keypoints.xy[0].cpu().numpy()  # shape (7, 2)
        conf = float(r.keypoints.conf[0].mean().cpu().numpy()) if r.keypoints.conf is not None else 0.0

        return PoseKeypoints(
            c7=tuple(kp[0]),
            left_shoulder=tuple(kp[1]),
            right_shoulder=tuple(kp[2]),
            curve_peak=tuple(kp[3]),
            left_pelvis=tuple(kp[4]),
            right_pelvis=tuple(kp[5]),
            sacral_base=tuple(kp[6]),
            confidence=conf,
        )

    def _mock_predict(self, image_bgr: np.ndarray) -> PoseKeypoints:
        """
        Deterministic placeholder used only when no trained weights exist.
        Produces a plausible, slightly-asymmetric skeleton centered on the
        image so downstream angle math and the API contract can be exercised
        end to end in dev/testing.
        """
        h, w = image_bgr.shape[:2] if image_bgr is not None else (640, 640)
        cx = w / 2
        return PoseKeypoints(
            c7=(cx, h * 0.15),
            left_shoulder=(cx - w * 0.12, h * 0.20),
            right_shoulder=(cx + w * 0.10, h * 0.19),
            curve_peak=(cx + w * 0.04, h * 0.45),
            left_pelvis=(cx - w * 0.10, h * 0.85),
            right_pelvis=(cx + w * 0.10, h * 0.85),
            sacral_base=(cx, h * 0.90),
            confidence=0.0,  # 0 confidence signals "mock, not a real prediction"
        )


# Module-level singleton so weights load once per process
pose_estimator = PoseEstimator()
