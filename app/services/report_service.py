from app.vision.pose_estimation import PoseKeypoints
from app.vision.classifier import ConditionResult


def build_clinical_summary(kp: PoseKeypoints, result: ConditionResult, shoulder_y_offset: float) -> str:
    if result.condition_tag == "Normal":
        return (
            "No significant structural curve was detected in this screening. "
            "Shoulder height difference measured "
            f"{abs(shoulder_y_offset):.1f} pixels, within the expected range."
        )

    side = "right" if shoulder_y_offset > 0 else "left"

    if "scoliosis" in result.condition_tag.lower():
        return (
            f"The AI detected a lateral spinal bend measuring {result.curve_degree}\u00b0 deviation. "
            f"Your {side} shoulder blade rests {abs(shoulder_y_offset):.1f} pixels higher on the "
            "vertical axis than the other side, indicating a possible structural curve shift. "
            "This is a screening estimate, not a diagnosis -- please confirm with an imaging-based "
            "Cobb angle measurement from a licensed clinician."
        )

    if result.condition_tag == "Kyphosis Deformity":
        return (
            f"The AI detected a thoracic curvature of approximately {result.curve_degree}\u00b0 on the "
            "sagittal profile, above the screening threshold for typical kyphotic rounding. "
            "Please confirm with a licensed clinician."
        )

    if result.condition_tag == "Hyperlordosis":
        return (
            f"The AI detected an increased lower-back curvature (variance: {result.curve_degree}) "
            "relative to the expected range for this profile. "
            "Please confirm with a licensed clinician."
        )

    return "Screening complete. Please review results with a licensed clinician."
