# MLOps v0.2

## Reproducible training

```bash
pip install -e '.[dev,tracking]'
python -m fraudml.train --n 12000 --algorithm auto
cat artifacts/metrics.json
```

`auto` compares Logistic Regression and XGBoost hyperparameters; for each classifier,
it compares hybrid anomaly weights 0/.2/.4 on the validation period. The capacity
threshold is chosen on validation scores, then frozen for one final chronological test.

The training dataset is synthetic and none of the test numbers imply business suitability.

## MLflow tracking and registration

MLflow is opt-in and uses the configured local SQLite tracking backend by default.
To log and create an immutable registered model version:

```bash
pip install -e '.[tracking,boosting]'
python -m fraudml.train --algorithm auto --register-model
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

Set `MLFLOW_TRACKING_URI` to an authorized remote tracking server when available.
`MLFLOW_MODEL_NAME` customizes the model name. Registration **does not promote** a
model or make it production-ready. No cloud MLflow server is provisioned by this repo.

## Operational signals

- `GET /metrics`: Prometheus request totals, latency histogram and review decisions.
- `GET /health`: process alive; `GET /ready`: model artifact available.
- `python scripts/check_drift.py`: reference-vs-current PSI and feature alert report.

PSI thresholds are configurable heuristics, **not validated production limits**.
Monitoring does not automatically collect user events, and no user data is sent externally.

## Container CI/CD

Every push runs tests and a Docker build. A published GitHub release or manual
workflow dispatch triggers image build and GHCR push to
`ghcr.io/OWNER/fraud-detection-ml-system`. It does **not** deploy to a live
service, run canaries, or provision cloud resources.

The image intentionally generates a demo model at build time. Before real deployment:
pin a validated model artifact by digest, sign artifacts, scan image, configure workload
identity/secrets, use a persistent model registry, and add approval gates.

## Limits and future architecture

No Kafka, feature store, Kubernetes, cloud deployment, SHAP, calibration or production
label pipeline is implemented yet. Public GitHub Actions and GHCR permissions can vary;
check workflow runs and package visibility before distributing an image.
