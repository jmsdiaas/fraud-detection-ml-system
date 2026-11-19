from fastapi.testclient import TestClient

from fraudml.api import app
from fraudml.data import make_transactions
from fraudml.monitor import drift_report


def test_metrics_endpoint():
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert "fraud_prediction_requests_total" in response.text


def test_drift_report():
    df = make_transactions(1200)
    report = drift_report(df.iloc[:600], df.iloc[600:])
    assert report["status"] in {"alert", "ok"}
    assert len(report["psi_by_feature"]) == 9
