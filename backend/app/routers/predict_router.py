"""
POST /api/predict

store_id is always current_user.store_id - never accepted from the request
body - so a user can only ever get predictions scoped to their own store.
"""
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User, InventoryData
from app.schemas import PredictRequest, PredictResponse
from app.ml.predict import predict_demand

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    history_rows = (
        db.query(InventoryData)
        .filter(
            InventoryData.store_id == current_user.store_id,
            InventoryData.product_id == payload.product_id,
        )
        .order_by(InventoryData.date.asc())
        .limit(60)
        .all()
    )
    history_df = pd.DataFrame([{"date": r.date, "demand": r.demand} for r in history_rows])

    try:
        result = predict_demand(current_user.store_id, payload.model_dump(), history_df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        store_id=current_user.store_id,
        product_id=payload.product_id,
        predicted_demand=result["predicted_demand"],
        inventory_level=payload.inventory_level,
        recommended_restock_qty=result["recommended_restock_qty"],
        model_version=result["model_version"],
    )
