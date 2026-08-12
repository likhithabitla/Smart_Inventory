"""
Centralized configuration for Smart Inventory Advisor.

All secrets and tunables are read from environment variables (.env file)
so the same codebase can run in dev / staging / prod without code changes.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Database -----------------------------------------------------
    database_url: str = "postgresql+psycopg2://inventory_user:inventory_pass@localhost:5432/smart_inventory"

    # --- Auth / JWT -----------------------------------------------------
    jwt_secret_key: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 480  # 8 hour shift

    # --- ML / model storage ----------------------------------------------
    model_dir: str = "./model_store"

    # Half-life (in days) for the exponential recency-decay weighting used
    # during training. lambda = ln(2) / half_life. See ml/weighting.py for
    # the full justification of this design choice.
    decay_half_life_days: float = 60.0

    # Minimum rows required in a store's data before /train is allowed.
    min_rows_to_train: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
