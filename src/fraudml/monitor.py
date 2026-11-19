"""Feature distribution drift: PSI and alerts from explicit reference/current windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURES


def psi(reference, current, bins: int = 10) -> float:
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if not len(ref) or not len(cur) or not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("Nonempty finite reference and current required")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(cur == ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    return {feature: psi(reference[feature], current[feature]) for feature in FEATURES}


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.2) -> dict:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    scores = feature_drift(reference, current)
    alerts = sorted(key for key, value in scores.items() if value >= threshold)
    return {"reference_rows": len(reference), "current_rows": len(current),
            "psi_threshold": threshold, "psi_by_feature": scores,
            "alert_features": alerts, "status": "alert" if alerts else "ok"}
