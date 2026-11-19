from fastapi.testclient import TestClient
from fraudml.release import build_release
from fraudml.secure_api import app, get_release

PAYLOAD = {"amount": 190, "hour": 2, "account_age_days": 12,
           "transactions_24h": 8, "avg_amount_30d": 30, "distance_km": 88,
           "device_new": 1, "ip_risk_score": .8, "merchant_risk_score": .7}

def test_auth_and_inference(tmp_path, monkeypatch):
    build_release(3000, output_dir=str(tmp_path))
    monkeypatch.setenv("FRAUD_RELEASE_DIR", str(tmp_path))
    monkeypatch.setenv("FRAUD_API_KEY", "unit-test-secret")
    get_release.cache_clear()
    client = TestClient(app)
    assert client.post("/v1/predict", json=PAYLOAD).status_code == 401
    response = client.post("/v1/predict", headers={"x-api-key": "unit-test-secret"}, json=PAYLOAD)
    assert response.status_code == 200
    assert 0 <= response.json()["calibrated_risk"] <= 1
    assert response.json()["dataset"] == "synthetic"
    assert client.post("/v1/predict", headers={"x-api-key": "unit-test-secret"},
                       json={**PAYLOAD, "hour": 50}).status_code == 422
    monkeypatch.setenv("FRAUD_REAL_TRANSACTIONS", "true")
    assert client.post("/v1/predict", headers={"x-api-key": "unit-test-secret"},
                       json=PAYLOAD).status_code == 403
    get_release.cache_clear()

def test_corrupt_artifact_fails_closed(tmp_path, monkeypatch):
    build_release(3000, output_dir=str(tmp_path))
    monkeypatch.setenv("FRAUD_RELEASE_DIR", str(tmp_path))
    monkeypatch.setenv("FRAUD_API_KEY", "unit-test-secret")
    (tmp_path / "model.joblib").write_bytes(b"tampered")
    get_release.cache_clear()
    assert TestClient(app).post("/v1/predict", headers={"x-api-key": "unit-test-secret"},
                                json=PAYLOAD).status_code == 503
    get_release.cache_clear()
