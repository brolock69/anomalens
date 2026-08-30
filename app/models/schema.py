from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# ---------- Stage 1: profile ----------
class UserProfileIn(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: Literal["male", "female", "other"]
    body_type: Optional[str] = None


class UserProfileOut(UserProfileIn):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- Stage 1: sensor gate (client reports these BEFORE upload) ----------
class CaptureConditions(BaseModel):
    lux: float
    pitch_deg: float
    roll_deg: float


class CaptureGateResult(BaseModel):
    ok: bool
    flash_recommended: bool
    reason: Optional[str] = None


# ---------- Stage 2: obstruction filter ----------
class ObstructionCheckResult(BaseModel):
    passed: bool
    edge_density: float
    reason: Optional[str] = None


# ---------- Stage 3/4: screening result ----------
class Landmark(BaseModel):
    x: float
    y: float


class ScreeningResult(BaseModel):
    curve_degree: float
    condition_tag: str
    condition_group: str
    confidence: float
    c7: Optional[Landmark] = None
    curve_peak: Optional[Landmark] = None
    sacral_base: Optional[Landmark] = None
    shoulder_y_offset: Optional[float] = None
    heatmap_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    clinical_summary: str


class ScreeningRecordOut(BaseModel):
    id: int
    timestamp: datetime
    curve_degree: float
    condition_tag: str
    condition_group: str
    confidence: Optional[float]
    clinical_summary: Optional[str]
    heatmap_path: Optional[str]

    class Config:
        from_attributes = True


class HistoryTrend(BaseModel):
    entries: list[ScreeningRecordOut]
    trend_direction: Literal["improving", "worsening", "stable", "insufficient_data"]


# ---------- Stage 4: clinics ----------
class ClinicOut(BaseModel):
    name: str
    specialty_group: str
    latitude: float
    longitude: float
    open_hour: int
    close_hour: int
    is_24_7: bool
    distance_km: Optional[float] = None
    is_open_now: Optional[bool] = None


class ClinicSearchByCoords(BaseModel):
    latitude: float
    longitude: float
    specialty_group: Optional[str] = None


class ClinicSearchByText(BaseModel):
    query: str
    specialty_group: Optional[str] = None
