"""Seeded, synthetic only; generated relationships are NOT empirical findings."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import FEATURES, TARGET, TIME


def make_transactions(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    if n < 300:
        raise ValueError("n must be at least 300")
    rng = np.random.default_rng(seed)
    time = pd.date_range("2025-01-01", periods=n, freq="10min", tz="UTC")
    amount = np.round(rng.lognormal(3.5, 1.05, n), 2)
    hour = np.asarray(time.hour)
    age = np.clip(rng.gamma(2.2, 175, n).astype(int), 0, 5000)
    count = rng.poisson(2.0, n)
    avg = np.maximum(3.0, rng.lognormal(3.6, .7, n))
    dist = rng.exponential(12, n)
    new = rng.binomial(1, .13, n)
    ip = rng.beta(1.5, 5, n)
    merchant = rng.beta(1.8, 5, n)
    score = (-7.2 + 1.4 * np.log1p(amount) / 5 + .22 * count
             + .017 * dist + 1.2 * new + 3.3 * ip
             + 3.2 * merchant - .002 * age
             + .5 * ((hour <= 4) | (hour >= 23)))
    probability = 1 / (1 + np.exp(-score))
    target = rng.binomial(1, probability)
    return pd.DataFrame({"transaction_id": [f"txn_{i:08d}" for i in range(n)],
        TIME: time, "amount": amount, "hour": hour, "account_age_days": age,
        "transactions_24h": count, "avg_amount_30d": avg, "distance_km": dist,
        "device_new": new, "ip_risk_score": ip, "merchant_risk_score": merchant,
        TARGET: target})


def validate_data(df: pd.DataFrame, require_target: bool = True) -> None:
    required = FEATURES + ([TARGET, TIME] if require_target else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Features contain NaN/inf")
    if (df["amount"] < 0).any() or (~df["device_new"].isin([0, 1])).any():
        raise ValueError("Invalid amount or device_new")
    if require_target and not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary")
