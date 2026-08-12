"""
Exponential recency-decay sample weighting.

WHY EXPONENTIAL DECAY (design rationale, also required by the spec):
Retail demand drift is continuous, not a step function - a promotion pattern
from 18 months ago is somewhat relevant, one from 3 years ago barely is, and
there's no natural cutoff where old data suddenly becomes "invalid". A hard
cutoff (e.g. "only use last 90 days") throws away useful signal and reacts
badly to stores with sparse upload history. Exponential decay instead gives
every row *some* weight, smoothly down-ranking older rows, so:
  - recent seasonal/promotional patterns dominate the loss function
  - old patterns are never fully forgotten (helpful for low-frequency SKUs)
  - the model adapts gracefully as new data arrives, without retraining logic
    needing to know about "windows" or "cutoffs"

FORMULA:
    weight = exp(-lambda * days_since_record)

CHOOSING LAMBDA VIA HALF-LIFE (much easier to reason about than a raw
lambda): we parameterize by "half-life" H = the number of days after which
a record's weight has decayed to 50%. Then:

    lambda = ln(2) / H

We default H = 60 days (configurable via DECAY_HALF_LIFE_DAYS). That means:
  - a record from today: weight ~= 1.0
  - a record from 60 days ago: weight ~= 0.5
  - a record from 120 days ago: weight ~= 0.25
  - a record from 1 year ago: weight ~= 0.006 (still nonzero, still counted)

60 days was chosen because retail promotion/season cycles typically run on a
4-8 week cadence; a half-life shorter than that would make the model overly
reactive to a single week's noise, while a much longer half-life would blunt
the benefit of decay entirely (approaching a flat/uniform weighting). Shops
with highly seasonal inventory can lower DECAY_HALF_LIFE_DAYS for a more
reactive model, or raise it for more stability on sparse data.
"""
import numpy as np
import pandas as pd


def compute_lambda(half_life_days: float) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    return np.log(2) / half_life_days


def compute_sample_weights(dates: pd.Series, reference_date, half_life_days: float) -> np.ndarray:
    """
    dates: pandas Series of datetime-like values (the record dates)
    reference_date: the "today" anchor (usually the most recent date in the
        training set, or datetime.utcnow()) that decay is measured from
    half_life_days: see module docstring

    Returns a numpy array of weights in (0, 1], same length/order as `dates`.
    """
    lam = compute_lambda(half_life_days)
    dates = pd.to_datetime(dates)
    reference_date = pd.to_datetime(reference_date)
    days_since = (reference_date - dates).dt.days.clip(lower=0).astype(float)
    weights = np.exp(-lam * days_since.values)
    return weights
