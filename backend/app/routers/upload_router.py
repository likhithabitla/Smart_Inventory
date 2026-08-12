"""
POST /api/upload/preview  - parse + validate + return a preview (no DB write)
POST /api/upload/commit   - append validated rows to inventory_data

Two-step flow matches the spec's "show preview before upload" requirement.
store_id is ALWAYS set from the authenticated user, never from the file or
any client-supplied field, so a store can never write data into another
store's rows.
"""
import io
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User, InventoryData
from app.schemas import UploadPreviewResponse, UploadCommitResponse

router = APIRouter(prefix="/upload", tags=["upload"])

REQUIRED_COLUMNS = ["date", "product_id"]
ALLOWED_COLUMNS = [
    "date", "product_id", "category", "region", "inventory_level",
    "units_sold", "units_ordered", "price", "discount", "weather_condition",
    "promotion", "competitor_pricing", "seasonality", "epidemic", "demand",
]

# simple in-memory cache of the last parsed-but-uncommitted upload per user,
# so /commit doesn't require re-uploading the file. In a multi-worker prod
# deployment this would move to Redis; documented as a follow-up in README.
_PENDING_UPLOADS = {}


def _read_file(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    filename = (file.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, .xls files are supported")


@router.post("/preview", response_model=UploadPreviewResponse)
def upload_preview(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    df = _read_file(file)
    df.columns = [c.strip().lower() for c in df.columns]

    warnings: List[str] = []
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_required}",
        )

    unknown_cols = [c for c in df.columns if c not in ALLOWED_COLUMNS]
    if unknown_cols:
        warnings.append(f"Ignoring unrecognized columns: {unknown_cols}")

    if "demand" not in df.columns:
        warnings.append(
            "No 'demand' column found - rows will be stored but excluded "
            "from model training until a demand value is provided."
        )

    _PENDING_UPLOADS[current_user.user_id] = df

    preview = df.head(10).fillna("").to_dict(orient="records")
    return UploadPreviewResponse(
        columns=[c for c in df.columns if c in ALLOWED_COLUMNS],
        row_count=len(df),
        preview_rows=preview,
        warnings=warnings,
    )


@router.post("/commit", response_model=UploadCommitResponse)
def upload_commit(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    df = _PENDING_UPLOADS.get(current_user.user_id)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No previewed upload found. Call /upload/preview first.",
        )

    inserted = 0
    for _, row in df.iterrows():
        record = InventoryData(
            date=pd.to_datetime(row.get("date")).date(),
            store_id=current_user.store_id,  # NEVER trust client-supplied store_id
            product_id=str(row.get("product_id")),
            category=row.get("category") if pd.notna(row.get("category")) else None,
            region=row.get("region") if pd.notna(row.get("region")) else None,
            inventory_level=_safe_float(row.get("inventory_level")),
            units_sold=_safe_float(row.get("units_sold")),
            units_ordered=_safe_float(row.get("units_ordered")),
            price=_safe_float(row.get("price")),
            discount=_safe_float(row.get("discount")),
            weather_condition=row.get("weather_condition") if pd.notna(row.get("weather_condition")) else None,
            promotion=int(row.get("promotion")) if pd.notna(row.get("promotion")) else 0,
            competitor_pricing=_safe_float(row.get("competitor_pricing")),
            seasonality=row.get("seasonality") if pd.notna(row.get("seasonality")) else None,
            epidemic=int(row.get("epidemic")) if pd.notna(row.get("epidemic")) else 0,
            demand=_safe_float(row.get("demand")),
        )
        db.add(record)
        inserted += 1

    db.commit()
    _PENDING_UPLOADS.pop(current_user.user_id, None)

    return UploadCommitResponse(inserted_rows=inserted, store_id=current_user.store_id)


def _safe_float(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None
