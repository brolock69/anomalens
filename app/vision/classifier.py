"""
Stage 4: ALGORITHMIC POST-PROCESSING DETERMINATION TREE.
Pure business-rule layer on top of the geometry computed in angle_calculator.
Deliberately separated from the math engine so thresholds can be tuned/
clinically reviewed without touching vector math.
"""
from dataclasses import dataclass
from app.config import settings
from app.vision.pose_estimation import PoseKeypoints
from app.vision import angle_calculator as calc


@dataclass
class ConditionResult:
    condition_tag: str
    condition_group: str
    curve_degree: float


def classify_coronal(kp: PoseKeypoints) -> ConditionResult:
    angle = calc.compute_coronal_angle(kp)

    if angle > settings.scoliosis_angle_threshold_deg and calc.is_lateral_curve(kp):
        direction = calc.curve_direction(kp)
        tag = "Levoscoliosis" if direction == "levo" else "Dextroscoliosis"
        return ConditionResult(condition_tag=tag, condition_group="Spine Surgery", curve_degree=angle)

    return ConditionResult(condition_tag="Normal", condition_group="None", curve_degree=angle)


def classify_sagittal(upper_kp: PoseKeypoints, lower_kp: PoseKeypoints) -> ConditionResult:
    upper_arc = calc.compute_sagittal_arc_angle(upper_kp, lower_kp)

    if upper_arc > settings.kyphosis_arc_threshold_deg:
        return ConditionResult(condition_tag="Kyphosis Deformity", condition_group="Spine Rehab", curve_degree=upper_arc)

    # NOTE: real lordosis scoring should come from lordosis_classifier_weights
    # (trained on the vertebral-column CSV) rather than this placeholder --
    # see services/report_service.py for where that model plugs in.
    lower_hollow_variance = abs(lower_kp.curve_peak[0] - lower_kp.sacral_base[0])
    age_group_tolerance = 25.0  # placeholder static tolerance; replace with
                                # age-bucketed lookup once you have normative data
    if lower_hollow_variance > age_group_tolerance:
        return ConditionResult(condition_tag="Hyperlordosis", condition_group="PT & Orthotics", curve_degree=lower_hollow_variance)

    return ConditionResult(condition_tag="Normal", condition_group="None", curve_degree=upper_arc)
