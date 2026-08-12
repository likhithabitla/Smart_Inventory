"""
Pydantic schemas for request validation and response serialization.
"""
from datetime import date as date_type, datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- Auth ----
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=6, max_length=128)
    store_id: str = Field(min_length=1, max_length=50)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    store_id: str
    username: str


# ---------------------------------------------------------- Inventory row -
class InventoryRow(BaseModel):
    date: date_type
    product_id: str
    category: Optional[str] = None
    region: Optional[str] = None
    inventory_level: Optional[float] = None
    units_sold: Optional[float] = None
    units_ordered: Optional[float] = None
    price: Optional[float] = None
    discount: Optional[float] = None
    weather_condition: Optional[str] = None
    promotion: Optional[int] = 0
    competitor_pricing: Optional[float] = None
    seasonality: Optional[str] = None
    epidemic: Optional[int] = 0
    demand: Optional[float] = None


class UploadPreviewResponse(BaseModel):
    columns: List[str]
    row_count: int
    preview_rows: List[dict]
    warnings: List[str] = []


class UploadCommitResponse(BaseModel):
    inserted_rows: int
    store_id: str


# ---------------------------------------------------------------- Train ---
class TrainResponse(BaseModel):
    status: str
    store_id: str
    model_version: str
    rows_used_total: int
    rows_used_this_store: int
    train_mae: float
    train_rmse: float
    trained_at: datetime
    lambda_decay: float
    half_life_days: float


# ------------------------------------------------------------- Predict ----
class PredictRequest(BaseModel):
    product_id: str
    date: date_type
    category: Optional[str] = None
    region: Optional[str] = None
    inventory_level: float = 0
    price: Optional[float] = None
    discount: Optional[float] = 0
    weather_condition: Optional[str] = None
    promotion: Optional[int] = 0
    competitor_pricing: Optional[float] = None
    seasonality: Optional[str] = None
    epidemic: Optional[int] = 0


class PredictResponse(BaseModel):
    store_id: str
    product_id: str
    predicted_demand: float
    inventory_level: float
    recommended_restock_qty: float
    model_version: str


# ------------------------------------------------------------ Dashboard ---
class DashboardDataResponse(BaseModel):
    store_id: str
    total_rows: int
    date_range: Optional[dict]
    sales_trend: List[dict]
    demand_trend: List[dict]
    top_products: List[dict]
    seasonality_breakdown: List[dict]
    weather_impact: List[dict]
    promotion_impact: List[dict]
    epidemic_impact: List[dict]
    inventory_vs_demand: List[dict]


class StoreSummaryResponse(BaseModel):
    store_id: str
    total_products: int
    total_records: int
    avg_units_sold: Optional[float]
    avg_inventory_level: Optional[float]
    last_trained_at: Optional[datetime]
    model_version: Optional[str]
