"""Real-time inference. Input validation; no raw transaction storage."""
from __future__ import annotations
import os
from functools import lru_cache
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(title="Fraud Detection ML System", version="0.1.0")


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(ge=0)
    transactions_24h: int = Field(ge=0)
    avg_amount_30d: float = Field(ge=0)
    distance_km: float = Field(ge=0)
    device_new: int = Field(ge=0, le=1)
    ip_risk_score: float = Field(ge=0, le=1)
    merchant_risk_score: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    supervised_score: float
    anomaly_score: float
    risk_score: float
    threshold: float
    review_required: bool
    model_version: str


@lru_cache(maxsize=1)
def load_model():
    path = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
    try:
        return joblib.load(path)  # Only load trusted locally-produced artifacts.
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"Model missing: {path}; train it first") from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    try:
        payload = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    scores = payload["model"].predict_components(pd.DataFrame([transaction.model_dump()])).iloc[0]
    return Prediction(supervised_score=float(scores.supervised_score),
         anomaly_score=float(scores.anomaly_score), risk_score=float(scores.risk_score),
         threshold=float(payload["threshold"]), review_required=bool(scores.risk_score >= payload["threshold"]),
         model_version=payload["model_version"])
