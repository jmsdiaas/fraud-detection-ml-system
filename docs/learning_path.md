# Technical walkthrough (senior DS -> ML engineer)

## Lesson 1 — Fraud framing
Rank high-risk card-not-present events for manual investigation, with limited review capacity. Why PR-AUC? Rare positives make ROC-AUC alone misleading. Exercise: change synthetic base rate and compare metrics.

## Lesson 2 — Data and leakage
Open `data.py`: columns are intended to be available at event time. Explain temporal splitting. Exercise: invent one leaky feature and explain offline score inflation.

## Lesson 3 — Supervised vs unsupervised
Open `model.py`: logistic regression learns labels and Isolation Forest models legitimate-data outliers. The two scores have different semantics. Exercise: compare anomaly weights 0, .2 and .4 on validation PR-AUC.

## Lesson 4 — Business policy
Open `evaluation.py`: select a quantile threshold on validation to target 5% review capacity. Exercise: try `--review-fraction 0.02`, quantify recall tradeoff.

## Lesson 5 — Serving and contracts
Start FastAPI and send a POST request. Follow Pydantic validation -> feature order -> joblib inference -> decision. Exercise: add a request_id for traceability without storing PII.

## Lesson 6 — Monitoring and MLOps
Run drift script; understand PSI distribution comparisons. Distinguish covariate drift, concept drift and delayed-label performance drift.

## Interview defense
Discuss class imbalance, selection bias from previous fraud rules, delayed labels, online/offline feature parity, false-positive costs, threshold governance, shadow deployment, alert fatigue, calibration and data privacy.
