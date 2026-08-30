"""
Stage 2: RUN INTERFERENCE AND NOISE EVALUATOR
Classical CV pass that runs BEFORE the image is ever sent to the deep model.
Cheap, fast, and lets us reject bad captures (loose clothing, hair overlay,
wrinkles) without burning a GPU inference call.

This is a heuristic filter, not a classifier -- it estimates whether the
back surface is "clean" enough for landmark detection using edge density.
Tune `obstruction_edge_density_max` in config.py against your own labeled
captures once you have real user data.
"""
import cv2
import numpy as np
from app.config import settings
from app.models.schema import ObstructionCheckResult


def check_obstruction(image_bgr: np.ndarray) -> ObstructionCheckResult:
    if image_bgr is None or image_bgr.size == 0:
        return ObstructionCheckResult(passed=False, edge_density=0.0, reason="empty_image")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, settings.canny_low_threshold, settings.canny_high_threshold)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # High edge density over a large area usually means clothing wrinkles/
    # patterned fabric/hair strands rather than a smooth skin surface with a
    # few clean bony-landmark edges (scapulae, spine groove, waist crease).
    if edge_density > settings.obstruction_edge_density_max:
        return ObstructionCheckResult(
            passed=False,
            edge_density=edge_density,
            reason="Obstruction detected: please wear tight athletic gear / tie hair up",
        )

    return ObstructionCheckResult(passed=True, edge_density=edge_density, reason=None)
