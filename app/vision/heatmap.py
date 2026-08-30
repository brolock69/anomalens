"""
GRAD-CAM HEAT MAP ENGINE + Stage 5 rendering (vector overlay + colormap).

Real Grad-CAM requires hooking into the trained model's final conv layer
(see `compute_gradcam` -- wire this to your ResNet/EfficientNet backbone
once it's trained). Until then, `render_overlay` still works standalone:
it draws the skeleton vector lines used in the "Interactive Historical
Progress Monitor" (Stage 5) directly from the pose keypoints, which is
useful for QA even before Grad-CAM is hooked up.
"""
import cv2
import numpy as np
from pathlib import Path
from app.vision.pose_estimation import PoseKeypoints
from app.config import settings


def compute_gradcam(model, image_tensor, target_layer):
    """
    Placeholder hook for real Grad-CAM once the classification backbone is
    trained. Wire this up with pytorch-grad-cam or a manual forward+backward
    hook on `target_layer`. Left unimplemented on purpose -- do not fabricate
    gradient data.
    """
    raise NotImplementedError(
        "Wire this to your trained ResNet/EfficientNet backbone's final conv "
        "layer once training is complete (see pytorch-grad-cam library)."
    )


def render_overlay(
    image_bgr: np.ndarray,
    kp: PoseKeypoints,
    heatmap_gray: np.ndarray | None,
    save_path: Path,
) -> Path:
    """Draws the vector analytics layer (Stage 5) and optional heatmap alpha blend."""
    canvas = image_bgr.copy()

    if heatmap_gray is not None:
        heat_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        canvas = cv2.addWeighted(canvas, 0.6, heat_color, 0.4, 0)

    pts = {
        "c7": tuple(map(int, kp.c7)),
        "curve_peak": tuple(map(int, kp.curve_peak)),
        "sacral_base": tuple(map(int, kp.sacral_base)),
    }
    cv2.line(canvas, pts["c7"], pts["curve_peak"], (0, 0, 255), 2)
    cv2.line(canvas, pts["curve_peak"], pts["sacral_base"], (0, 0, 255), 2)
    for p in pts.values():
        cv2.circle(canvas, p, 5, (0, 0, 255), -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
    return save_path
