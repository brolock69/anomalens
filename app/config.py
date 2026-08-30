"""
Central configuration for the AnomaLens backend.
All tunable thresholds from the pipeline diagram live here so they can be
adjusted without touching business logic.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "AnomaLens API"
    api_prefix: str = "/api/v1"
    debug: bool = True

    # --- Storage ---
    database_url: str = f"sqlite:///{BASE_DIR}/data/anomalens.db"
    cache_dir: Path = BASE_DIR / "cache"
    weights_dir: Path = BASE_DIR / "weights"

    # --- Capture / pre-processing thresholds (from pipeline Stage 1-2) ---
    low_light_lux_threshold: float = 15.0
    gyro_pitch_target_deg: float = 90.0
    gyro_roll_target_deg: float = 0.0
    gyro_tolerance_deg: float = 5.0
    capture_image_size: int = 640

    # --- Obstruction filter thresholds (Stage 2) ---
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    obstruction_edge_density_max: float = 0.35  # heuristic cutoff

    # --- Model weight file names (Stage 3) ---
    pose_model_weights: str = "pose_keypoints_best.pt"       # YOLOv8-pose, trained on
                                                              # posture-types-image-dataset
    scoliosis_seg_weights: str = "scoliosis_seg_best.pt"     # trained on scoliosis-yes/no sets
    kyphosis_weights: str = "kyphosis_best.pt"               # trained on roboflow kyphosis sets
    lordosis_classifier_weights: str = "lordosis_clf.joblib" # trained on vertebral-column csv

    # --- Diagnostic thresholds (Stage 4) ---
    scoliosis_angle_threshold_deg: float = 10.0
    kyphosis_arc_threshold_deg: float = 45.0
    hyperlordosis_variance_key: str = "age_group_default"

    # --- Clinics / geolocation ---
    clinics_data_path: Path = BASE_DIR / "data" / "clinics.json"
    default_region: str = "Zamboanga City"

    # --- History ---
    history_lookback_entries: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.weights_dir.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
