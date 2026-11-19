"""Hybrid supervised/anomaly model with train-only preprocessing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from .schema import FEATURES


class HybridFraudModel:
    def __init__(self, anomaly_weight: float = .20, seed: int = 42):
        if not 0 <= anomaly_weight <= 1:
            raise ValueError("anomaly_weight must be in [0, 1]")
        self.anomaly_weight = anomaly_weight
        self.seed = seed
        self.preprocessor = Pipeline([("impute", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())])
        self.classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        self.detector = IsolationForest(n_estimators=120, contamination="auto", random_state=seed, n_jobs=-1)
        self.reference_anomalies = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(np.unique(y)) != 2:
            raise ValueError("Both classes required to train")
        values = self.preprocessor.fit_transform(X[FEATURES])
        self.classifier.fit(values, y)
        legit = values[np.asarray(y) == 0]
        self.detector.fit(legit)
        self.reference_anomalies = np.sort(-self.detector.score_samples(legit))
        return self

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.reference_anomalies is None:
            raise RuntimeError("Model has not been fitted")
        values = self.preprocessor.transform(X[FEATURES])
        supervised = self.classifier.predict_proba(values)[:, 1]
        raw_anomaly = -self.detector.score_samples(values)
        anomaly = np.searchsorted(self.reference_anomalies, raw_anomaly, side="right") / len(self.reference_anomalies)
        hybrid = (1 - self.anomaly_weight) * supervised + self.anomaly_weight * anomaly
        return pd.DataFrame({"supervised_score": supervised, "anomaly_score": anomaly,
                             "risk_score": hybrid}, index=X.index)
