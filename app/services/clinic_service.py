import json
from app.config import settings
from app.models.schema import ClinicOut
from app.utils.geo import haversine_km, is_open_now


def _load_clinics() -> list[dict]:
    with open(settings.clinics_data_path) as f:
        return json.load(f)


def find_by_coords(lat: float, lon: float, specialty_group: str | None = None) -> list[ClinicOut]:
    clinics = _load_clinics()
    results = []
    for c in clinics:
        if specialty_group and c["specialty_group"] != specialty_group:
            continue
        distance = haversine_km(lat, lon, c["latitude"], c["longitude"])
        results.append(
            ClinicOut(
                name=c["name"],
                specialty_group=c["specialty_group"],
                latitude=c["latitude"],
                longitude=c["longitude"],
                open_hour=c["open_hour"],
                close_hour=c["close_hour"],
                is_24_7=c["is_24_7"],
                distance_km=distance,
                is_open_now=is_open_now(c["open_hour"], c["close_hour"], c["is_24_7"]),
            )
        )
    results.sort(key=lambda r: r.distance_km)
    return results


def find_by_region_text(query: str, specialty_group: str | None = None) -> list[ClinicOut]:
    clinics = _load_clinics()
    query_lower = query.strip().lower()
    results = []
    for c in clinics:
        if specialty_group and c["specialty_group"] != specialty_group:
            continue
        if query_lower in c.get("region", "").lower() or query_lower in c["name"].lower():
            results.append(
                ClinicOut(
                    name=c["name"],
                    specialty_group=c["specialty_group"],
                    latitude=c["latitude"],
                    longitude=c["longitude"],
                    open_hour=c["open_hour"],
                    close_hour=c["close_hour"],
                    is_24_7=c["is_24_7"],
                    is_open_now=is_open_now(c["open_hour"], c["close_hour"], c["is_24_7"]),
                )
            )
    return results
