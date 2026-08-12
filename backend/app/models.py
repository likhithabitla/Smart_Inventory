"""
ORM models. Mirrors schema.sql exactly - schema.sql is the source of truth
for the raw DDL (useful for manual psql setup / migrations), these classes
are what the application code uses at runtime.
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, Boolean,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    store_id = Column(String(50), nullable=False, index=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    inventory_rows = relationship("InventoryData", back_populates="owner_store", viewonly=True,
                                   primaryjoin="foreign(InventoryData.store_id)==User.store_id")


class InventoryData(Base):
    """
    Raw uploaded inventory / sales records. store_id is ALWAYS taken from the
    authenticated user's JWT on write - it is never trusted from client input.
    Every read query in this app filters on store_id (see routers/*.py) to
    guarantee tenant isolation at the application layer, in addition to the
    DB index below which makes that filter cheap.
    """
    __tablename__ = "inventory_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    store_id = Column(String(50), nullable=False, index=True)
    product_id = Column(String(50), nullable=False, index=True)
    category = Column(String(100))
    region = Column(String(100))
    inventory_level = Column(Float)
    units_sold = Column(Float)
    units_ordered = Column(Float)
    price = Column(Float)
    discount = Column(Float)
    weather_condition = Column(String(50))
    promotion = Column(Integer)          # 0/1 flag
    competitor_pricing = Column(Float)
    seasonality = Column(String(50))
    epidemic = Column(Integer)           # 0/1 flag
    demand = Column(Float)               # target variable
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    owner_store = relationship("User", viewonly=True,
                                primaryjoin="foreign(InventoryData.store_id)==User.store_id")

    __table_args__ = (
        Index("ix_inventory_store_product_date", "store_id", "product_id", "date"),
    )


class ModelMetadata(Base):
    """
    Tracks the latest trained model version per store (or 'GLOBAL' for the
    shared model file). See ml/train_model.py for why one global model is
    used instead of one model per store.
    """
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(String(50), nullable=False, index=True)
    last_trained_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50), nullable=False)
    rows_used = Column(Integer)
    train_mae = Column(Float)
    train_rmse = Column(Float)

    __table_args__ = (
        UniqueConstraint("store_id", "model_version", name="uq_store_model_version"),
    )
