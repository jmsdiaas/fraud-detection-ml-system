"""Population Stability Index; reference bins derived only from reference data."""
import numpy as np
import pandas as pd
from .schema import FEATURES


def psi(reference, current, bins=10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0 or not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("Nonempty finite reference and current required")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(cur == ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c-r) * np.log(c/r)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    return {feature: psi(reference[feature], current[feature]) for feature in FEATURES}
