# Smart Inventory Advisor

A multi-tenant demand forecasting and restock recommendation system for shop
owners. Upload sales history, train a shared ML model, predict demand per
product, and view store-scoped analytics dashboards.

## Folder structure

```
smart-inventory-advisor/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entrypoint
│   │   ├── config.py                # Settings (env vars)
│   │   ├── database.py              # SQLAlchemy engine/session
│   │   ├── models.py                # ORM models (users, inventory_data, model_metadata)
│   │   ├── schemas.py               # Pydantic request/response schemas
│   │   ├── security.py              # bcrypt hashing + JWT encode/decode
│   │   ├── auth.py                  # get_current_user dependency (tenant isolation)
│   │   ├── routers/
│   │   │   ├── auth_router.py       # POST /register, /login
│   │   │   ├── upload_router.py     # POST /upload/preview, /upload/commit
│   │   │   ├── train_router.py      # POST /train
│   │   │   ├── predict_router.py    # POST /predict
│   │   │   └── dashboard_router.py  # GET /dashboard-data, /store-summary
│   │   └── ml/
│   │       ├── features.py          # Feature engineering + lag features (leakage-safe)
│   │       ├── weighting.py         # Exponential recency-decay sample weights
│   │       ├── train_model.py       # LightGBM training + persistence
│   │       └── predict.py           # Inference + restock calculation
│   ├── model_store/                 # Trained model artifacts (.joblib) — created at runtime
│   ├── requirements.txt
│   ├── .env.example
│   └── schema.sql                   # Authoritative PostgreSQL DDL
├── frontend/
│   ├── index.html                   # Home / marketing page
│   ├── login.html
│   ├── register.html
│   ├── upload.html                  # Preview → confirm upload flow
│   ├── predict.html                 # Prediction form + result
│   ├── dashboard.html                # Chart.js analytics
│   ├── css/style.css
│   └── js/api.js                    # Shared fetch client, auth/session helpers
└── sample_data/
    └── sample_inventory.csv          # Example dataset you can upload immediately
```

## 1. Prerequisites

- Python 3.10+
- PostgreSQL 13+ running locally or reachable over the network
- pip

## 2. Database setup

Create the database and load the schema:

```bash
createdb smart_inventory
psql -d smart_inventory -f backend/schema.sql
```

(Alternatively, `Base.metadata.create_all()` runs automatically on backend
startup for local/dev convenience — `schema.sql` remains the source of truth
for production migrations.)

## 3. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env: set DATABASE_URL to match your Postgres instance,
# and set JWT_SECRET_KEY to a long random string.

uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000/api`, with interactive docs at
`http://localhost:8000/docs`.

## 4. Frontend

The FastAPI app automatically serves `frontend/` as static files at
`http://localhost:8000/`, so with the backend running you can just open:

```
http://localhost:8000/
```

To serve the frontend separately instead (e.g. via a static host or `python
-m http.server` from the `frontend/` folder), open `frontend/js/api.js` and
change `API_BASE` to your backend's full URL, then serve the `frontend/`
directory with any static file server.

## 5. Try it end to end

1. Open the app → **Create your store account** → pick a `store_id` (e.g.
   `store-04`), a username, and a password.
2. Go to **Upload** → drag in `sample_data/sample_inventory.csv` → review the
   preview → **Confirm & append to store data**.
3. Click **Train model now** (needs ≥30 rows in your store by default —
   adjustable via `MIN_ROWS_TO_TRAIN` if you edit `config.py`).
4. Go to **Predict** → enter a `product_id` from the sample data (e.g.
   `SKU-001`), fill in the conditions, and get a predicted demand + restock
   quantity.
5. Go to **Dashboard** to see trends, top products, and the effect of
   weather/promotions/epidemics on demand for your store.

Register a second account with a different `store_id` to confirm that
store's dashboard, uploads, and predictions are completely separate.

## 6. Dataset format

CSV or Excel with these columns (header names are case-insensitive):

| Column               | Required | Notes                                   |
|-----------------------|----------|------------------------------------------|
| date                  | Yes      | `YYYY-MM-DD`                              |
| product_id            | Yes      | e.g. `SKU-001`                            |
| category              | No       | e.g. `Beverages`                          |
| region                | No       | e.g. `North`                              |
| inventory_level       | No       | units currently on shelf/in stock         |
| units_sold            | No       | actual units sold that day                |
| units_ordered         | No       | units ordered/replenished that day        |
| price                 | No       | selling price                             |
| discount              | No       | discount applied (%)                      |
| weather_condition     | No       | e.g. `Sunny`, `Rainy`                     |
| promotion             | No       | `0` or `1`                                 |
| competitor_pricing    | No       | nearby competitor's price                 |
| seasonality           | No       | e.g. `Winter`, `Holiday`                  |
| epidemic              | No       | `0` or `1`, disruption flag               |
| demand                | Recommended | historical demand — **required for training**; rows without it are stored but skipped when training |

Unrecognized columns are ignored (with a warning shown in the preview).
`store_id` is never read from the file — it is always taken from your logged
-in session, so it's impossible to upload into another store's data.

## 7. Key design decisions

- **One global model, not one per store.** New stores have little or no
  data of their own; a shared LightGBM model with `store_id`/`product_id` as
  categorical features transfers cross-store patterns immediately while
  still learning store-specific effects through those identity features.
  Full rationale is in `backend/app/ml/features.py`.
- **Exponential recency-decay weighting.** `weight = exp(-λ × days_since)`,
  with λ derived from a configurable half-life (default 60 days). Recent
  sales dominate the loss while older data is never fully discarded. Full
  rationale is in `backend/app/ml/weighting.py`.
- **Leakage-safe lag features.** `demand_lag_1`, `demand_lag_7`,
  `rolling_mean_7`, `rolling_mean_30` are computed per `(store_id,
  product_id)` using only values strictly before the current row's date
  (`shift(1)` before any rolling window) — see `build_lag_features()` in
  `backend/app/ml/features.py`.
- **Tenant isolation.** `store_id` is only ever read from the authenticated
  JWT (`get_current_user` in `backend/app/auth.py`), never from client
  input, and every database query in every router filters on it explicitly.

## 8. Production notes / follow-ups

- The `_PENDING_UPLOADS` in-memory cache in `upload_router.py` (used between
  preview and commit) should move to Redis or similar for multi-worker
  deployments.
- Tighten CORS `allow_origins` in `main.py` to your actual frontend origin(s).
- Consider a background job queue (Celery/RQ) if training time grows with
  data volume, so `/train` doesn't block a web worker.
- Add rate limiting on `/login` and `/register`.
