from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import settings
from app.models import db_models
from app.models.schema import UserProfileIn, UserProfileOut, HistoryTrend, ScreeningRecordOut

router = APIRouter(tags=["users & history"])


@router.post("/users", response_model=UserProfileOut)
def create_user(profile: UserProfileIn, db: Session = Depends(get_db)):
    user = db_models.User(age=profile.age, sex=profile.sex, body_type=profile.body_type)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/users/{user_id}", response_model=UserProfileOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(db_models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/users/{user_id}/history", response_model=HistoryTrend)
def get_history(user_id: int, db: Session = Depends(get_db)):
    """
    Stage 5: USER REGISTRATION PROGRESSIVITY ENGINE.
    Reads back the last N entries and reports whether the trend line of
    curve_degree is improving, worsening, or stable.
    """
    records = (
        db.query(db_models.ScreeningRecord)
        .filter(db_models.ScreeningRecord.user_id == user_id)
        .order_by(db_models.ScreeningRecord.timestamp.desc())
        .limit(settings.history_lookback_entries)
        .all()
    )

    if len(records) < 2:
        trend = "insufficient_data"
    else:
        # records are newest-first; compare newest to oldest in the window
        newest, oldest = records[0].curve_degree, records[-1].curve_degree
        if newest < oldest - 1.0:
            trend = "improving"
        elif newest > oldest + 1.0:
            trend = "worsening"
        else:
            trend = "stable"

    return HistoryTrend(
        entries=[ScreeningRecordOut.model_validate(r) for r in records],
        trend_direction=trend,
    )
