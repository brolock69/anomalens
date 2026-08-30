"""
DIFFERENTIAL MATH ENGINE.
Pure geometry -- no ML here, so it's fully unit-testable and doesn't depend
on model weights being trained yet.

Computes a 2D surface angle equivalent to a Cobb angle from three landmarks,
plus the shoulder-height offset used in the plain-language report.
"""
import math
from app.vision.pose_estimation import PoseKeypoints


def _vector(p_from: tuple[float, float], p_to: tuple[float, float]) -> tuple[float, float]:
    return (p_to[0] - p_from[0], p_to[1] - p_from[1])


def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_theta))


def _vertex_deviation(c7: tuple[float, float], peak: tuple[float, float], base: tuple[float, float]) -> float:
    """
    Vertex-angle convention: both vectors originate AT the curve peak and
    point out to the two endpoints (peak->c7, peak->base). For a perfectly
    straight spine, the peak sits on the line between c7 and base, so the
    two vectors point in opposite directions and the angle between them is
    ~180 deg. We report the deviation from straight (180 - angle), so
    0 deg = straight and larger values = more curvature.
    """
    v1 = _vector(peak, c7)
    v2 = _vector(peak, base)
    raw_angle = _angle_between(v1, v2)
    return round(abs(180.0 - raw_angle), 2)


def compute_coronal_angle(kp: PoseKeypoints) -> float:
    """Surface-level equivalent of a Cobb angle for the coronal (back) view."""
    return _vertex_deviation(kp.c7, kp.curve_peak, kp.sacral_base)


def compute_shoulder_y_offset(kp: PoseKeypoints) -> float:
    """Positive = right shoulder sits higher (smaller y) than left, and vice versa."""
    return round(kp.left_shoulder[1] - kp.right_shoulder[1], 2)


def compute_sagittal_arc_angle(upper_kp: PoseKeypoints, lower_kp: PoseKeypoints) -> float:
    """
    The upper and lower sagittal-profile frames are two separate photos, each
    with its own C7/curve_peak/sacral_base triple in its own frame of
    reference -- their coordinates aren't directly comparable to each other.
    So we compute the vertex deviation independently within each frame and
    average the two, rather than mixing coordinates across frames.
    """
    upper_dev = _vertex_deviation(upper_kp.c7, upper_kp.curve_peak, upper_kp.sacral_base)
    lower_dev = _vertex_deviation(lower_kp.c7, lower_kp.curve_peak, lower_kp.sacral_base)
    return round((upper_dev + lower_dev) / 2.0, 2)


def is_lateral_curve(kp: PoseKeypoints) -> bool:
    """True if curve_peak x is offset from the C7->sacral_base midline enough
    to indicate a real lateral bend rather than noise."""
    midline_x = (kp.c7[0] + kp.sacral_base[0]) / 2
    return abs(kp.curve_peak[0] - midline_x) > 1e-3


def curve_direction(kp: PoseKeypoints) -> str:
    """'levo' (bends left, peak left of neck) or 'dextro' (bends right)."""
    return "levo" if kp.curve_peak[0] < kp.c7[0] else "dextro"
