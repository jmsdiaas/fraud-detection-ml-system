# Production-readiness assessment (v0.4)

**Status: NOT APPROVED for real payment traffic.** This project deliberately uses synthetic data and cannot certify fraud performance, privacy, cost economics or operational suitability.

## Improvements actually implemented
- Five chronological partitions: training 55%, model selection 15%, score calibration 10%, business decision policy 10%, untouched final test 10%.
- Platt calibration is fitted independently of selection and decision policy, with a test-only evaluation.
- Calibrated risk and frozen review threshold are served by an authenticated /v1/predict endpoint.
- Strict Pydantic schema and explicit refusal to run with FRAUD_REAL_TRANSACTIONS=true.
- Joblib model checksum verified before loading and versioned release manifest.
- Separate demo container (Dockerfile.release); credentials provided at runtime, never committed.

## Local reproducibility
```bash
pip install -e '.[dev,boosting]'
python -m fraudml.release --n 12000 --algorithm auto
export FRAUD_API_KEY='locally-generated-strong-secret'
uvicorn fraudml.secure_api:app --host 127.0.0.1 --port 8000
pytest -q
```

To build reproducible synthetic image:
```bash
docker build -f Dockerfile.release -t fraud-release-demo .
docker run --rm -e FRAUD_API_KEY=example-local-secret -p 127.0.0.1:8000:8000 fraud-release-demo
```

Never expose the demo container publicly, and rotate credentials if compromised. The API key is a lightweight demonstration safeguard, NOT enterprise IAM. For real deployment use private networking, OAuth/JWT validation or service-to-service identity, quotas, audit trails, network restrictions and secrets manager.

## Required before any live decision automation
1. Lawful, representative, point-in-time feature data and mature fraud labels; privacy impact assessment, LGPD controls, minimization, access governance and retention.
2. Segment-aware performance and stability study across time windows, bootstrap CIs, rare-fraud PR-AUC, recall at review capacity, business cost with approved loss/recovery rates, proper calibration and feature/label drift monitoring.
3. Independent review and approval of decision thresholds, human-in-the-loop fallback, dispute and appeal handling, bias/fairness and model risk reviews.
4. Reproducible signed model/image provenance, artifact verification (hash is not a signature), SBOM, vulnerability scanning, pinned dependencies, secret rotation, penetration testing, rate limiting and authenticated monitoring.
5. Production observability (SLOs, tracing, error budgets, drift alerts), shadow runs, load tests, canary and rollback, incident response playbooks and backup/recovery.
6. Authorized cloud project, billing controls, Secret Manager, network/IAM and staging approval. Cloud Run workflow is a *template*; no live cloud installation has occurred.

## Operational release gate
No synthetic demo may claim 'production-ready'. Promotion requires signed approval documents and real-data measurements. A README or passing unit tests are insufficient.
