"""Supervised-classifier explanations; do not confuse contributions with causal effects."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import FEATURES

def explain_supervised(model, transactions: pd.DataFrame) -> dict:
    x = model.preprocessor.transform(transactions[FEATURES])
    if getattr(model, "classifier_kind", "logistic") == "xgboost":
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install SHAP for TreeSHAP explanations") from exc
        values = shap.TreeExplainer(model.classifier)(x)
        contributions = np.asarray(values.values)
        baseline = np.asarray(values.base_values)
        method, unit = "tree_shap", "raw_model_margin"
    else:
        contributions = x * model.classifier.coef_[0]
        baseline = np.full(len(x), float(model.classifier.intercept_[0]))
        method, unit = "linear_logit_contributions", "log_odds"
    return {"method": method, "unit": unit, "features": FEATURES,
            "baseline": np.asarray(baseline).reshape(-1).tolist(),
            "contributions": contributions.tolist(),
            "warning": "Explains supervised model only, not hybrid score or calibrated probability."}
