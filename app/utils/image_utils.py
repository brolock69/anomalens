import cv2
import numpy as np
from pathlib import Path
from fastapi import UploadFile


async def read_upload_as_bgr(file: UploadFile) -> np.ndarray:
    """Decode an UploadFile into an OpenCV BGR array."""
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {file.filename}")
    return image


def downscale(image_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)


def clear_cache_dir(cache_dir: Path) -> None:
    """
    Stage 5: HARDWARE DATA PRIVACY SANITIZATION AND CLEANUP LOOP.
    Wipes raw capture frames after a screening completes. Annotated
    report images/heatmaps that the user chose to save live under a
    separate 'reports/' subfolder and are NOT touched by this.
    """
    for item in cache_dir.glob("*"):
        if item.is_file():
            item.unlink(missing_ok=True)
