"""
POST /api/train

Any authenticated store owner can trigger a retrain. Because this app uses
ONE global model (see ml/train_model.py docstring), the retrain uses ALL
rows currently in inventory_data across every store - the triggering
store's own new uploads are included, and the resulting model also benefits
every other store. Per-store isolation is preserved at PREDICTION time
(store_id is always taken from the JWT) and in the metadata record we save
per store, not by training a separate model per tenant.
"""
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import User, InventoryData, ModelMetadata
from app.schemas import TrainResponse
from app.ml.train_model import train_global_model

router = APIRouter(tags=["train"])


@router.post("/train", response_model=TrainResponse)
def train(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    store_row_count = db.query(InventoryData).filter(
        InventoryData.store_id == current_user.store_id
    ).count()

    if store_row_count < settings.min_rows_to_train:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Your store has only {store_row_count} uploaded rows. "
                f"Upload at least {settings.min_rows_to_train} rows before training."
            ),
        )

    all_rows = db.query(InventoryData).all()
    df = pd.DataFrame([{
        "date": r.date, "store_id": r.store_id, "product_id": r.product_id,
        "category": r.category, "region": r.region,
        "inventory_level": r.inventory_level, "units_sold": r.units_sold,
        "units_ordered": r.units_ordered, "price": r.price, "discount": r.discount,
        "weather_condition": r.weather_condition, "promotion": r.promotion,
        "competitor_pricing": r.competitor_pricing, "seasonality": r.seasonality,
        "epidemic": r.epidemic, "demand": r.demand,
    } for r in all_rows])

    try:
        result = train_global_model(df, triggering_store_id=current_user.store_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    metadata = ModelMetadata(
        store_id=current_user.store_id,
        last_trained_at=result["trained_at"],
        model_version=result["model_version"],
        rows_used=result["rows_used_this_store"],
        train_mae=result["train_mae"],
        train_rmse=result["train_rmse"],
    )
    db.add(metadata)
    db.commit()

    return TrainResponse(status="success", store_id=current_user.store_id, **result)
