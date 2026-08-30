# AnomaLens Backend

FastAPI backend for the AnomaLens AI-based posture/scoliosis self-screening
app, implementing the pipeline from your architecture diagram:

1. **Capture gating** (`/screening/capture-gate`) — light/gyro checks before the client snaps a photo.
2. **Obstruction filter** — classical CV (Canny edge density) rejects loose clothing/hair-covered frames before they hit the model.
3. **Pose estimation** — YOLOv8-pose wrapper for 7 landmarks (C7, shoulders, curve peak, pelvis, sacral base).
4. **Angle math + condition tagging** — pure-geometry Cobb-angle-equivalent calculation, then rule-based tagging (Levo/Dextroscoliosis, Kyphosis, Hyperlordosis).
5. **Reporting** — plain-language clinical summary, annotated overlay image, and per-user history/trend tracking.
6. **Clinics** — GPS-based (Haversine) and text-based clinic/hospital lookup with open-now status.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate          # venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

## Plugging in your trained models

The backend runs today in **mock mode** — `PoseEstimator` returns a
deterministic placeholder skeleton whenever no weights file is found, so you
can build/test the mobile app against a live API before training finishes.

To go live:

1. Train your YOLOv8-pose model on the keypoints dataset (`posture-types-image-dataset-frontback-view` / your `posture-keypoints` notebook), exporting these 7 keypoints in this exact order:
   `C7, left_shoulder, right_shoulder, curve_peak, left_pelvis, right_pelvis, sacral_base`.
2. Drop the exported `.pt` file into `/weights/pose_keypoints_best.pt` (filename configurable in `app/config.py`).
3. Restart the API — `PoseEstimator` auto-detects the weights file and switches out of mock mode.
4. For Grad-CAM heatmaps, wire `app/vision/heatmap.py::compute_gradcam` to your trained ResNet/EfficientNet backbone's final conv layer (e.g. via `pytorch-grad-cam`) — this is left as a `NotImplementedError` stub on purpose since fabricating gradient data would be meaningless.
5. For lordosis, the vertebral-column CSV dataset is tabular, not image-based, so it plugs in as a separate scikit-learn classifier (`lordosis_classifier_weights`) rather than through the pose model — wire it into `app/vision/classifier.py::classify_sagittal` where the placeholder heuristic currently sits.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/users` | Create a user profile (age, sex, body_type) |
| GET | `/api/v1/users/{id}` | Fetch profile |
| GET | `/api/v1/users/{id}/history` | Last N screenings + trend direction |
| POST | `/api/v1/screening/capture-gate` | Pre-capture sensor check (flash/tilt) |
| POST | `/api/v1/screening/analyze` | Upload 3 frames, run full pipeline, get diagnosis |
| POST | `/api/v1/clinics/nearby` | GPS-based clinic search (Haversine) |
| POST | `/api/v1/clinics/search` | Text-based municipality/province search |

`/screening/analyze` expects `multipart/form-data`:
- `user_id` (form field)
- `coronal_image`, `sagittal_upper_image`, `sagittal_lower_image` (files)

## Important: this is a screening aid, not a diagnostic device

Every clinical summary explicitly states this is a screening estimate. Angle
values are a 2D surface-landmark approximation, not a radiographic Cobb
angle — keep that framing in the mobile UI and any user-facing copy, and
route flagged users to an actual clinician rather than presenting the
app's output as a diagnosis. Worth reviewing your congress paper's validity/
usability findings against these thresholds before clinical claims.

## Config

All thresholds (lux cutoff, gyro tolerance, scoliosis/kyphosis angle
thresholds, edge-density cutoff for the obstruction filter, clinic data
path) live in `app/config.py` — tune them there rather than in code, ideally
against your own labeled validation set once you have one.

## Data model

SQLite by default (`data/anomalens.db`), via SQLAlchemy — swap
`database_url` in `app/config.py` for Postgres/MySQL in production without
touching any router code.

## Project layout

```
app/
  main.py              FastAPI app + router registration
  config.py            All tunable settings
  database.py          SQLAlchemy engine/session
  models/
    db_models.py        ORM: User, ScreeningRecord (patient_history)
    schema.py            Pydantic request/response contracts
  routers/
    screening.py         capture-gate + analyze
    history.py            users + history/trend
    clinics.py             nearby (GPS) + search (text)
  vision/
    obstruction_filter.py  Stage 2: classical CV clothing/hair filter
    pose_estimation.py     Stage 3: YOLOv8-pose wrapper (+ mock fallback)
    angle_calculator.py     Pure geometry: vertex-angle Cobb-equivalent math
    classifier.py            Stage 4: rule-based condition tagging
    heatmap.py                Grad-CAM hook (stub) + overlay rendering
  services/
    clinic_service.py        Haversine distance + open-hours logic
    report_service.py         Plain-language clinical summary strings
  utils/
    geo.py, image_utils.py
data/
  clinics.json            Sample clinic directory (edit/replace with real data)
weights/                  Drop your trained .pt/.joblib files here
```
