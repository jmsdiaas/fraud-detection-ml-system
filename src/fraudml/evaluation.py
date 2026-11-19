"""Cost-sensitive evaluation and capacity-based thresholds."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, confusion_matrix


def threshold_for_review_capacity(scores, review_fraction: float = .05) -> float:
    if not 0 < review_fraction < 1:
        raise ValueError("review_fraction must be between zero and one")
    return float(np.quantile(np.asarray(scores), 1 - review_fraction))


def evaluate(y, scores, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "review_rate": float(pred.mean()), "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn), "threshold": float(threshold)}
