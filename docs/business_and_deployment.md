# Calibration, financial decisions, explanations and deployment

The analysis uses synthetic data and assumed costs. Results cannot be generalized to actual fraud outcomes.

## Run
```bash
pip install -e '.[dev,explain]'
python -m fraudml.train --n 12000 --seed 42 --algorithm auto
python scripts/evaluate_business.py --n 12000 --seed 42 --capacity .05
python scripts/explain_sample.py
docker compose up --build
```

Default illustrative unit costs: 2 per manual review, 5 friction per false positive, residual fraud loss after 85% prevention for reviewed fraud, 100% loss for missed fraud. Manual review threshold is constrained to 5% on validation. The test rate can differ.

The model-selection process in v0.2 already used the full validation sample. This follow-on post-hoc experiment splits that same period into calibration and policy halves; evaluation on the test period is untouched. **An independent future validation window is needed for unbiased deployment policy assessment.** The calibrated score and financial threshold are evaluated offline and are not yet wired to the live API.

Explanations: TreeSHAP for XGBoost, additive log-odds contributions for logistic regression. Both explain the supervised component only, not hybrid risk or calibrated probability.

Prometheus available at http://localhost:9090 and FastAPI at http://localhost:8000 when using Docker Compose. This local stack has no persistent monitoring storage or alerts.

The Cloud Run workflow is manual, restricted to GitHub environment 'staging', and requires project setup and GitHub variables: GCP_PROJECT_ID, GCP_REGION, GCP_WIF_PROVIDER, GCP_SERVICE_ACCOUNT, GCP_RUNTIME_SERVICE_ACCOUNT. Create Artifact Registry 'fraud-ml', configure WIF permissions, workload identity, billing limits, staging environment required reviewers, and monitor resource cost before triggering. The service disallows unauthenticated requests and limits to one instance. Code publication is **not** a live cloud deployment. Its container uses a synthetic demo model built into the image. Do not send real payment transactions or personal data.
