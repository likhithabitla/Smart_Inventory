"""
GET /api/dashboard-data - full analytics payload for the logged-in store
GET /api/store-summary  - lightweight header/summary stats

Every query below filters on InventoryData.store_id == current_user.store_id.
This is the enforcement point for "each store sees ONLY their data" on the
read side (the write side is enforced in upload_router.py).
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.auth import get_current_user
from app.database import get_db
from app.models import User, InventoryData, ModelMetadata
from app.schemas import DashboardDataResponse, StoreSummaryResponse

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard-data", response_model=DashboardDataResponse)
def dashboard_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(InventoryData).filter(InventoryData.store_id == current_user.store_id)
    rows = q.all()

    if not rows:
        return DashboardDataResponse(
            store_id=current_user.store_id, total_rows=0, date_range=None,
            sales_trend=[], demand_trend=[], top_products=[],
            seasonality_breakdown=[], weather_impact=[], promotion_impact=[],
            epidemic_impact=[], inventory_vs_demand=[],
        )

    import pandas as pd
    df = pd.DataFrame([{
        "date": r.date, "product_id": r.product_id, "category": r.category,
        "units_sold": r.units_sold, "demand": r.demand,
        "inventory_level": r.inventory_level, "weather_condition": r.weather_condition,
        "promotion": r.promotion, "epidemic": r.epidemic, "seasonality": r.seasonality,
    } for r in rows])

    sales_trend = (
        df.groupby("date")["units_sold"].sum().reset_index()
        .sort_values("date").rename(columns={"units_sold": "value"})
    )
    demand_trend = (
        df.groupby("date")["demand"].sum().reset_index()
        .sort_values("date").rename(columns={"demand": "value"})
    )
    top_products = (
        df.groupby("product_id")["units_sold"].sum().reset_index()
        .sort_values("units_sold", ascending=False).head(10)
        .rename(columns={"units_sold": "value"})
    )
    seasonality_breakdown = (
        df.dropna(subset=["seasonality"]).groupby("seasonality")["demand"].mean().reset_index()
        .rename(columns={"demand": "avg_demand"})
    )
    weather_impact = (
        df.dropna(subset=["weather_condition"]).groupby("weather_condition")["demand"].mean().reset_index()
        .rename(columns={"demand": "avg_demand"})
    )
    promotion_impact = (
        df.groupby("promotion")["demand"].mean().reset_index()
        .rename(columns={"demand": "avg_demand", "promotion": "promotion_flag"})
    )
    epidemic_impact = (
        df.groupby("epidemic")["demand"].mean().reset_index()
        .rename(columns={"demand": "avg_demand", "epidemic": "epidemic_flag"})
    )
    inventory_vs_demand = (
        df.groupby("date")[["inventory_level", "demand"]].mean().reset_index()
        .sort_values("date")
    )

    def to_records(frame):
        return frame.fillna(0).astype(object).where(pd.notnull(frame), None).to_dict(orient="records")

    return DashboardDataResponse(
        store_id=current_user.store_id,
        total_rows=len(df),
        date_range={"start": str(df["date"].min()), "end": str(df["date"].max())},
        sales_trend=_serialize_dates(sales_trend),
        demand_trend=_serialize_dates(demand_trend),
        top_products=top_products.to_dict(orient="records"),
        seasonality_breakdown=seasonality_breakdown.to_dict(orient="records"),
        weather_impact=weather_impact.to_dict(orient="records"),
        promotion_impact=promotion_impact.to_dict(orient="records"),
        epidemic_impact=epidemic_impact.to_dict(orient="records"),
        inventory_vs_demand=_serialize_dates(inventory_vs_demand),
    )


def _serialize_dates(frame):
    frame = frame.copy()
    frame["date"] = frame["date"].astype(str)
    return frame.to_dict(orient="records")


@router.get("/store-summary", response_model=StoreSummaryResponse)
def store_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(InventoryData).filter(InventoryData.store_id == current_user.store_id)

    total_records = q.count()
    total_products = (
        db.query(func.count(func.distinct(InventoryData.product_id)))
        .filter(InventoryData.store_id == current_user.store_id)
        .scalar()
    )
    avg_units_sold = (
        db.query(func.avg(InventoryData.units_sold))
        .filter(InventoryData.store_id == current_user.store_id)
        .scalar()
    )
    avg_inventory_level = (
        db.query(func.avg(InventoryData.inventory_level))
        .filter(InventoryData.store_id == current_user.store_id)
        .scalar()
    )

    latest_meta = (
        db.query(ModelMetadata)
        .filter(ModelMetadata.store_id == current_user.store_id)
        .order_by(ModelMetadata.last_trained_at.desc())
        .first()
    )

    return StoreSummaryResponse(
        store_id=current_user.store_id,
        total_products=total_products or 0,
        total_records=total_records,
        avg_units_sold=float(avg_units_sold) if avg_units_sold is not None else None,
        avg_inventory_level=float(avg_inventory_level) if avg_inventory_level is not None else None,
        last_trained_at=latest_meta.last_trained_at if latest_meta else None,
        model_version=latest_meta.model_version if latest_meta else None,
    )
