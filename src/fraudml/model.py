"""Leakage-safe hybrid of supervised learning and training-only anomaly baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import FEATURES


class HybridFraudModel:
    def __init__(self, anomaly_weight: float = 0.2, seed: int = 42,
                 classifier_kind: str = "logistic", classifier_params: dict | None = None):
        if not 0 <= anomaly_weight <= 1:
            raise ValueError("anomaly_weight must be in [0, 1]")
        if classifier_kind not in {"logistic", "xgboost"}:
            raise ValueError("classifier_kind must be logistic or xgboost")
        self.anomaly_weight = anomaly_weight
        self.seed = seed
        self.classifier_kind = classifier_kind
        self.classifier_params = classifier_params or {}
        self.preprocessor = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        self.classifier = None
        self.detector = IsolationForest(n_estimators=120, contamination="auto", random_state=seed, n_jobs=1)
        self.reference_anomalies = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        labels = np.asarray(y)
        if len(np.unique(labels)) != 2:
            raise ValueError("Both classes required to train")
        values = self.preprocessor.fit_transform(X[FEATURES])
        if self.classifier_kind == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError("Install with pip install -e '.[boosting]'") from exc
            neg = int((labels == 0).sum())
            pos = int((labels == 1).sum())
            params = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
                      "subsample": 0.85, "colsample_bytree": 0.9, "n_jobs": 1,
                      "tree_method": "hist", "random_state": self.seed, "eval_metric": "logloss",
                      "scale_pos_weight": neg / max(pos, 1)}
            params.update(self.classifier_params)
            self.classifier = XGBClassifier(**params)
        else:
            self.classifier = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=self.seed,
                **self.classifier_params,
            )
        self.classifier.fit(values, labels)
        genuine = values[labels == 0]
        self.detector.fit(genuine)
        self.reference_anomalies = np.sort(-self.detector.score_samples(genuine))
        return self

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.reference_anomalies is None:
            raise RuntimeError("Model has not been fitted")
        values = self.preprocessor.transform(X[FEATURES])
        supervised = self.classifier.predict_proba(values)[:, 1]
        raw_anomaly = -self.detector.score_samples(values)
        anomaly = np.searchsorted(self.reference_anomalies, raw_anomaly, side="right") / len(self.reference_anomalies)
        hybrid = (1 - self.anomaly_weight) * supervised + self.anomaly_weight * anomaly
        return pd.DataFrame({
            "supervised_score": supervised, "anomaly_score": anomaly, "risk_score": hybrid,
        }, index=X.index)
