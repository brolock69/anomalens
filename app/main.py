from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.models import db_models  # noqa: F401 -- ensures models are registered before create_all
from app.routers import screening, history, clinics

Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name)

# Loosen this in production to your actual mobile app origin / API gateway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(screening.router, prefix=settings.api_prefix)
app.include_router(history.router, prefix=settings.api_prefix)
app.include_router(clinics.router, prefix=settings.api_prefix)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}
