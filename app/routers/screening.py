import uuid
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import settings
from app.models import db_models
from app.models.schema import (
    CaptureConditions, CaptureGateResult, ScreeningResult, Landmark,
)
from app.utils.image_utils import read_upload_as_bgr, downscale, clear_cache_dir
from app.vision.obstruction_filter import check_obstruction
from app.vision.pose_estimation import pose_estimator
from app.vision import angle_calculator as calc
from app.vision import classifier
from app.vision.heatmap import render_overlay
from app.services.report_service import build_clinical_summary

router = APIRouter(prefix="/screening", tags=["screening"])


@router.post("/capture-gate", response_model=CaptureGateResult)
def capture_gate(conditions: CaptureConditions):
    """
    Stage 1: CAMERA PRE-CAPTURE CONTROLLER & HARDWARE TRIPPERS.
    Client calls this with live sensor readings BEFORE snapping the photo,
    so we can tell it to flip on the flashlight / tell the user to hold the
    phone flat, without wasting a capture.
    """
    flash_needed = conditions.lux < settings.low_light_lux_threshold

    pitch_ok = abs(conditions.pitch_deg - settings.gyro_pitch_target_deg) <= settings.gyro_tolerance_deg
    roll_ok = abs(conditions.roll_deg - settings.gyro_roll_target_deg) <= settings.gyro_tolerance_deg

    if not (pitch_ok and roll_ok):
        return CaptureGateResult(ok=False, flash_recommended=flash_needed, reason="Tilt phone flat")

    return CaptureGateResult(ok=True, flash_recommended=flash_needed, reason=None)


@router.post("/analyze", response_model=ScreeningResult)
async def analyze_screening(
    user_id: int = Form(...),
    coronal_image: UploadFile = File(..., description="Frame 1: Coronal back plane view"),
    sagittal_upper_image: UploadFile = File(..., description="Frame 2: Upper sagittal profile"),
    sagittal_lower_image: UploadFile = File(..., description="Frame 3: Lower sagittal profile"),
    db: Session = Depends(get_db),
):
    """
    Runs the full Stage 2-5 pipeline against the three captured frames and
    persists the result to patient_history.
    """
    user = db.query(db_models.User).filter(db_models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User profile not found. Create one via /users first.")

    coronal_bgr = downscale(await read_upload_as_bgr(coronal_image), settings.capture_image_size)
    upper_bgr = downscale(await read_upload_as_bgr(sagittal_upper_image), settings.capture_image_size)
    lower_bgr = downscale(await read_upload_as_bgr(sagittal_lower_image), settings.capture_image_size)

    # --- Stage 2: obstruction filter (all three frames must pass) ---
    for label, img in [("coronal", coronal_bgr), ("sagittal_upper", upper_bgr), ("sagittal_lower", lower_bgr)]:
        obstruction = check_obstruction(img)
        if not obstruction.passed:
            raise HTTPException(status_code=422, detail=f"{label}: {obstruction.reason}")

    # --- Stage 3: pose estimation on each frame ---
    coronal_kp = pose_estimator.predict(coronal_bgr)
    upper_kp = pose_estimator.predict(upper_bgr)
    lower_kp = pose_estimator.predict(lower_bgr)

    # --- Stage 3/4: classify coronal (scoliosis) and sagittal (kyphosis/lordosis) ---
    coronal_result = classifier.classify_coronal(coronal_kp)
    sagittal_result = classifier.classify_sagittal(upper_kp, lower_kp)

    # Prefer whichever result is non-normal; if both flag something, report the
    # coronal (scoliosis) finding first since it's routed to Spine Surgery.
    primary = coronal_result if coronal_result.condition_tag != "Normal" else sagittal_result

    shoulder_offset = calc.compute_shoulder_y_offset(coronal_kp)
    clinical_summary = build_clinical_summary(coronal_kp, primary, shoulder_offset)

    # --- Stage 5: render overlay + save ---
    report_id = uuid.uuid4().hex[:12]
    overlay_path = settings.cache_dir / "reports" / f"{report_id}_annotated.jpg"
    render_overlay(coronal_bgr, coronal_kp, heatmap_gray=None, save_path=overlay_path)

    record = db_models.ScreeningRecord(
        user_id=user_id,
        curve_degree=primary.curve_degree,
        condition_tag=primary.condition_tag,
        condition_group=primary.condition_group,
        confidence=coronal_kp.confidence,
        c7_x=coronal_kp.c7[0], c7_y=coronal_kp.c7[1],
        curve_peak_x=coronal_kp.curve_peak[0], curve_peak_y=coronal_kp.curve_peak[1],
        sacral_base_x=coronal_kp.sacral_base[0], sacral_base_y=coronal_kp.sacral_base[1],
        shoulder_y_offset=shoulder_offset,
        heatmap_path=str(overlay_path),
        annotated_image_path=str(overlay_path),
        clinical_summary=clinical_summary,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Stage 5: wipe raw capture cache (not the saved report) after processing
    clear_cache_dir(settings.cache_dir)

    return ScreeningResult(
        curve_degree=primary.curve_degree,
        condition_tag=primary.condition_tag,
        condition_group=primary.condition_group,
        confidence=coronal_kp.confidence,
        c7=Landmark(x=coronal_kp.c7[0], y=coronal_kp.c7[1]),
        curve_peak=Landmark(x=coronal_kp.curve_peak[0], y=coronal_kp.curve_peak[1]),
        sacral_base=Landmark(x=coronal_kp.sacral_base[0], y=coronal_kp.sacral_base[1]),
        shoulder_y_offset=shoulder_offset,
        heatmap_path=str(overlay_path),
        annotated_image_path=str(overlay_path),
        clinical_summary=clinical_summary,
    )
