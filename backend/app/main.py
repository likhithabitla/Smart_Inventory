"""
Smart Inventory Advisor - FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.routers import auth_router, upload_router, train_router, predict_router, dashboard_router

# Create tables if they don't exist yet (schema.sql is the authoritative
# source for manual DB setup; this is a convenience for local/dev use).
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Inventory Advisor API",
    description="Multi-tenant inventory demand forecasting & restock recommendation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router, prefix="/api")
app.include_router(upload_router.router, prefix="/api")
app.include_router(train_router.router, prefix="/api")
app.include_router(predict_router.router, prefix="/api")
app.include_router(dashboard_router.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the vanilla frontend directly from FastAPI for simple local setups.
# In production you would typically serve /frontend via nginx/CDN instead.
import os

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend")
try:
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
except RuntimeError:
    # frontend/ not present (e.g. running backend tests in isolation)
    pass
