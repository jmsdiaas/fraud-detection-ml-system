# Fraud Detection ML System

**v0.4 | Synthetic fraud detection reference system** combining supervised classification, anomaly detection, temporal validation and ML engineering. Educational prototype, **not a production fraud engine**.

**PRODUCTION GATE: Not approved for real payment traffic.** The trained model is synthetic and serving is demo-only. See [production readiness](docs/production_readiness.md).

## Features

- Reproducible synthetic transactional dataset, seeded with approximately 2% fraud.
- Chronological train/validation/test split (60/20/20); train-only transformations.
- Logistic regression and optional XGBoost, mixed with Isolation Forest; model selection on validation PR-AUC.
- Manual-review-capacity threshold, test-set confusion matrix, recall, precision, PR-AUC and ROC-AUC.
- FastAPI scoring, strict request schema, health/readiness endpoints.
- PSI drift reports, Prometheus metrics, pytest, Actions CI, Docker, MLflow registry and optional GHCR publishing.
- Offline Platt calibration, cost-constrained review policy and supervised-model SHAP explanations.
- Local Prometheus stack with Docker Compose; manual, private Cloud Run staging workflow (requires credentials).
- Architecture decisions, limitations and technical learning guide.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
python -m pip install -e '.[dev,explain,tracking]'
python -m fraudml.train --n 12000 --review-fraction 0.05 --algorithm auto
python scripts/evaluate_business.py --n 12000 --seed 42
python scripts/explain_sample.py
pytest -q
uvicorn fraudml.api:app --reload
```

Open http://127.0.0.1:8000/docs. Send a sample:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"amount":190,"hour":2,"account_age_days":12,"transactions_24h":8,"avg_amount_30d":30,"distance_km":88,"device_new":1,"ip_risk_score":0.8,"merchant_risk_score":0.7}'
```

## Release candidate: calibrated and authenticated demo

```bash
python -m fraudml.release --n 12000 --algorithm auto
export FRAUD_API_KEY='generate-a-local-secret'
uvicorn fraudml.secure_api:app --host 127.0.0.1 --port 8000
```

Use `POST /v1/predict` with the `x-api-key` header. Model artifact integrity is checked against the release manifest. The production-mode switch explicitly blocks real transactions.

## Docker

```bash
docker build -t fraud-detection-ml-system .
docker run --rm -p 8000:8000 fraud-detection-ml-system
```

The Docker image trains a **synthetic demo model at build time**. In production, deployment should load a separately validated, immutable model artifact.

## Testing

```bash
ruff check src tests scripts --select E4,E7,E9
pytest -q
python scripts/check_drift.py
```

## Project layout

- `src/fraudml/` — dataset, schemas, hybrid model, evaluation, training, API and drift
- `tests/` — deterministic data, model, monitoring, training and HTTP tests
- `scripts/` — local drift check
- `docs/` — architecture and technical lessons
- `.github/workflows/ci.yml` — lint and test workflow

## Model limitations

- Entirely synthetic data: scores are **not** real-world model performance.
- Risk scores are **not calibrated fraud probabilities**.
- No feature store or event streaming; MLflow registration is optional, and no cloud deployment has been executed.
- Offline calibrated policy is not wired into live API; API risk scores remain uncalibrated.
- The model artifact uses pickle-based serialization: load only trusted artifacts.
- Label maturity, fraud investigations, operational cost and governance are left as future upgrades.

See [architecture](docs/architecture.md), [MLOps guide](docs/mlops.md), [business policy and staging deployment](docs/business_and_deployment.md), and [learning path](docs/learning_path.md).

## Roadmap

1. Integrate separately validated calibration/decision policy into serving, plus SHAP risk-explanation API with access controls.
2. Point-in-time features and delayed fraud labels.
3. Data lineage, release approval gates and immutable runtime artifacts.
4. Production metrics, monitoring, reproducible infrastructure and canary deployment.

MIT License.
