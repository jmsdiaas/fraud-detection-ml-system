from fastapi.testclient import TestClient
from fraudml.api import app, load_model
from fraudml.train import train


def test_api_prediction(monkeypatch, tmp_path):
    train(n=3000, output_dir=str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    load_model.cache_clear()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    payload = {"amount": 190, "hour": 2, "account_age_days": 12,
        "transactions_24h": 8, "avg_amount_30d": 30, "distance_km": 88,
        "device_new": 1, "ip_risk_score": .8, "merchant_risk_score": .7}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert isinstance(body["review_required"], bool)
    payload["hour"] = 40
    assert client.post("/predict", json=payload).status_code == 422
    load_model.cache_clear()
