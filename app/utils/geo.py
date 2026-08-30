import math
from datetime import datetime, timezone


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius, km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return round(R * c, 2)


def is_open_now(open_hour: int, close_hour: int, is_24_7: bool, now: datetime | None = None) -> bool:
    if is_24_7:
        return True
    now = now or datetime.now(timezone.utc)
    current_hour = now.hour
    if open_hour <= close_hour:
        return open_hour <= current_hour < close_hour
    # handles clinics that wrap past midnight
    return current_hour >= open_hour or current_hour < close_hour
