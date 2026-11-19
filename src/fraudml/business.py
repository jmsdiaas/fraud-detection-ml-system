"""Post-hoc calibration and monetary policy on synthetic data; not production economics."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

@dataclass(frozen=True)
class Costs:
    review_cost: float = 2.0
    false_positive_friction: float = 5.0
    fraud_loss_multiplier: float = 1.0
    prevented_fraction: float = .85

    def __post_init__(self):
        if min(self.review_cost, self.false_positive_friction, self.fraud_loss_multiplier) < 0:
            raise ValueError("Costs must be nonnegative")
        if not 0 <= self.prevented_fraction <= 1:
            raise ValueError("prevented_fraction must be within [0, 1]")

class ScoreCalibrator:
    """Platt scaling trained on validation scores, never the chronological test."""
    def __init__(self):
        self.estimator = LogisticRegression(max_iter=1000)

    def fit(self, scores, labels):
        x, y = np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int)
        if len(x) != len(y) or len(np.unique(y)) != 2:
            raise ValueError("Calibration needs aligned scores and both classes")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite scores")
        self.estimator.fit(x, y)
        return self

    def predict_proba(self, scores):
        return self.estimator.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]

def total_cost(labels, amounts, scores, threshold: float, costs: Costs) -> dict:
    y, amt, s = np.asarray(labels, dtype=int), np.asarray(amounts, dtype=float), np.asarray(scores, dtype=float)
    if not (len(y) == len(amt) == len(s)) or not len(y):
        raise ValueError("Aligned nonempty arrays required")
    if not np.isfinite(s).all() or not np.isfinite(amt).all() or (amt < 0).any():
        raise ValueError("Nonfinite scores or invalid amounts")
    review = s >= threshold
    fraud = y == 1
    flagged = review & fraud
    missed = ~review & fraud
    false_positives = review & ~fraud
    operations = float(review.sum() * costs.review_cost)
    friction = float(false_positives.sum() * costs.false_positive_friction)
    residual = float((amt[flagged] * costs.fraud_loss_multiplier * (1 - costs.prevented_fraction)).sum())
    missed_loss = float((amt[missed] * costs.fraud_loss_multiplier).sum())
    return {"total_cost": operations + friction + residual + missed_loss,
            "review_count": int(review.sum()), "review_rate": float(review.mean()),
            "true_positives": int(flagged.sum()), "false_positives": int(false_positives.sum()),
            "missed_frauds": int(missed.sum()), "operational_cost": operations,
            "friction_cost": friction, "residual_fraud_loss": residual, "missed_fraud_loss": missed_loss}

def optimize_policy(labels, amounts, scores, costs: Costs, max_review_fraction: float = .05) -> dict:
    if not 0 < max_review_fraction <= 1:
        raise ValueError("max_review_fraction must be in (0, 1]")
    s = np.asarray(scores, dtype=float)
    thresholds = [float(np.nextafter(s.max(), np.inf))]
    thresholds += [float(t) for t in np.unique(np.quantile(
        s, np.linspace(max(0.0, 1 - max_review_fraction), 1, 101)))]
    candidates = []
    for threshold in thresholds:
        row = total_cost(labels, amounts, s, threshold, costs)
        if row["review_rate"] <= max_review_fraction + 1e-12:
            candidates.append({"threshold": threshold, **row})
    if not candidates:
        raise RuntimeError("No eligible threshold")
    return min(candidates, key=lambda row: (row["total_cost"], row["review_count"]))

def calibration_metrics(labels, raw_scores, calibrated_scores) -> dict:
    y = np.asarray(labels, dtype=int)
    return {"brier_raw": float(brier_score_loss(y, raw_scores)),
            "brier_calibrated": float(brier_score_loss(y, calibrated_scores)),
            "pr_auc": float(average_precision_score(y, calibrated_scores))}

def costs_dict(costs: Costs) -> dict:
    return asdict(costs)
