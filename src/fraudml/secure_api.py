"""Authenticated release scoring API. Explicitly rejects real-transaction use."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Fraud Detection Release API", version="0.4.0")

class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    amount: float = Field(ge=0, le=1e9)
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(ge=0, le=50000)
    transactions_24h: int = Field(ge=0, le=100000)
    avg_amount_30d: float = Field(ge=0, le=1e9)
    distance_km: float = Field(ge=0, le=50000)
    device_new: int = Field(ge=0, le=1)
    ip_risk_score: float = Field(ge=0, le=1)
    merchant_risk_score: float = Field(ge=0, le=1)

class Prediction(BaseModel):
    calibrated_risk: float
    decision: str
    threshold: float
    model_version: str
    dataset: str

def authorize(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("FRAUD_API_KEY")
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Authentication required")

@lru_cache(maxsize=1)
def get_release():
    directory = Path(os.getenv("FRAUD_RELEASE_DIR", "artifacts/release"))
    report = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    binary = directory / "model.joblib"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != report["model_sha256"]:
        raise RuntimeError("Release checksum mismatch")
    # Integrity does not imply authenticity: only load artifacts from trusted build pipelines.
    bundle = joblib.load(binary)
    if bundle.get("dataset") != "synthetic" or bundle.get("approved_for_real_transactions"):
        raise RuntimeError("Invalid demo model release metadata")
    return bundle

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready(_: None = Depends(authorize)):
    try:
        get_release()
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Release unavailable") from exc
    return {"status": "ready", "mode": "synthetic_demo"}

@app.post("/v1/predict", response_model=Prediction)
def predict(transaction: Transaction, _: None = Depends(authorize)):
    if os.getenv("FRAUD_REAL_TRANSACTIONS", "false").lower() == "true":
        raise HTTPException(status_code=403, detail="Real transactions not authorized")
    try:
        bundle = get_release()
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Release unavailable") from exc
    raw = bundle["model"].predict_components(pd.DataFrame([transaction.model_dump()])).risk_score.to_numpy()
    prob = float(bundle["calibrator"].predict_proba(raw)[0])
    return Prediction(calibrated_risk=prob,
                      decision="manual_review" if prob >= bundle["threshold"] else "pass",
                      threshold=float(bundle["threshold"]), model_version=bundle["model_version"],
                      dataset="synthetic")
