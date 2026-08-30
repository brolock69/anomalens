from fastapi import APIRouter
from app.models.schema import ClinicSearchByCoords, ClinicSearchByText, ClinicOut
from app.services import clinic_service

router = APIRouter(prefix="/clinics", tags=["clinics"])


@router.post("/nearby", response_model=list[ClinicOut])
def clinics_nearby(payload: ClinicSearchByCoords):
    """CHOICE 1 (GPS pathway): distances via Haversine, sorted nearest first."""
    return clinic_service.find_by_coords(payload.latitude, payload.longitude, payload.specialty_group)


@router.post("/search", response_model=list[ClinicOut])
def clinics_search(payload: ClinicSearchByText):
    """CHOICE 2 (manual lookup pathway): text match against municipality/province/name."""
    return clinic_service.find_by_region_text(payload.query, payload.specialty_group)
