"""
ORM models. Mirrors the pipeline's "Load Historical Matrix Loop" and
"User Registration Progressivity Engine" (Stage 1 & 5).
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)          # user_sex
    body_type = Column(String, nullable=True)     # body_type enum-ish
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    history = relationship("ScreeningRecord", back_populates="user", cascade="all, delete-orphan")


class ScreeningRecord(Base):
    """One row per screening run. Maps to `patient_history` in the diagram."""
    __tablename__ = "patient_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Core diagnostic outputs
    curve_degree = Column(Float, nullable=False)          # generic "curve_degree" from diagram
    condition_tag = Column(String, nullable=False)        # Levoscoliosis / Dextroscoliosis / Kyphosis Deformity / Hyperlordosis / Normal
    condition_group = Column(String, nullable=False)      # Spine Surgery / Spine Rehab / PT & Orthotics / None
    confidence = Column(Float, nullable=True)

    # Landmark coordinates (Stage 3, Branch 1)
    c7_x = Column(Float, nullable=True)
    c7_y = Column(Float, nullable=True)
    curve_peak_x = Column(Float, nullable=True)
    curve_peak_y = Column(Float, nullable=True)
    sacral_base_x = Column(Float, nullable=True)
    sacral_base_y = Column(Float, nullable=True)
    shoulder_y_offset = Column(Float, nullable=True)

    # Artifact paths
    heatmap_path = Column(String, nullable=True)
    annotated_image_path = Column(String, nullable=True)

    clinical_summary = Column(String, nullable=True)

    user = relationship("User", back_populates="history")
