-- =========================================================
-- Smart Inventory Advisor - PostgreSQL Schema
-- =========================================================
-- Multi-tenant design: every row that carries business data
-- is tagged with a store_id (free-text identifier chosen at
-- registration, e.g. "store-04"). ALL application queries
-- filter by store_id — enforced in the ORM layer
-- (backend/app/auth.py + every router), never trusted from
-- client input. This file is the authoritative DDL; the
-- SQLAlchemy models in app/models.py mirror it exactly.
-- =========================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -------------------------------------------------------
-- USERS
-- Each user (shop owner / staff) belongs to EXACTLY one
-- store_id. Anyone who registers with the same store_id
-- joins that store and shares its data — this is the
-- tenant boundary for the whole application.
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id         SERIAL PRIMARY KEY,
    username        VARCHAR(100) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    is_admin        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_store_id ON users(store_id);

-- -------------------------------------------------------
-- INVENTORY_DATA
-- The raw time series records uploaded by each store, plus
-- the historical demand target used for training/prediction.
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS inventory_data (
    id                      BIGSERIAL PRIMARY KEY,
    date                    DATE NOT NULL,
    store_id                VARCHAR(50) NOT NULL,
    product_id              VARCHAR(50) NOT NULL,
    category                VARCHAR(100),
    region                  VARCHAR(100),
    inventory_level         DOUBLE PRECISION,
    units_sold              DOUBLE PRECISION,
    units_ordered           DOUBLE PRECISION,
    price                   DOUBLE PRECISION,
    discount                DOUBLE PRECISION,
    weather_condition       VARCHAR(50),
    promotion               INTEGER DEFAULT 0,     -- 0/1 flag
    competitor_pricing      DOUBLE PRECISION,
    seasonality             VARCHAR(50),
    epidemic                INTEGER DEFAULT 0,     -- 0/1 flag
    demand                  DOUBLE PRECISION,      -- target variable (historical demand)
    uploaded_at             TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Critical composite index: every query filters by store_id
-- first, then usually product_id + date range for lag/rolling
-- feature computation.
CREATE INDEX IF NOT EXISTS idx_inventory_store_id ON inventory_data(store_id);
CREATE INDEX IF NOT EXISTS ix_inventory_store_product_date
    ON inventory_data(store_id, product_id, date);
CREATE INDEX IF NOT EXISTS idx_inventory_date ON inventory_data(date);

-- -------------------------------------------------------
-- MODEL_METADATA
-- Tracks training runs of the single GLOBAL model. A row is
-- written every time a store triggers /train, recording the
-- model version that was produced and how it performed.
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_metadata (
    id                  SERIAL PRIMARY KEY,
    store_id            VARCHAR(50) NOT NULL,
    last_trained_at     TIMESTAMP,
    model_version       VARCHAR(50) NOT NULL,
    rows_used           INTEGER,
    train_mae           DOUBLE PRECISION,
    train_rmse          DOUBLE PRECISION,
    CONSTRAINT uq_store_model_version UNIQUE (store_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_model_metadata_store_id ON model_metadata(store_id);
