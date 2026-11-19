"""Model selection on validation only; held-out temporal test evaluated once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from .data import make_transactions, validate_data
from .evaluation import evaluate, threshold_for_review_capacity
from .model import HybridFraudModel
from .schema import FEATURES, TARGET, TIME


def _candidate_models(seed: int, algorithm: str):
    kinds = ["logistic", "xgboost"] if algorithm == "auto" else [algorithm]
    for kind in kinds:
        params_grid = ([{}, {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.04}]
                       if kind == "xgboost" else [{}])
        for params in params_grid:
            yield kind, params


def train(n: int = 12000, seed: int = 42, output_dir: str = "artifacts",
          review_fraction: float = 0.05, algorithm: str = "logistic",
          register_model: bool = False) -> dict:
    if algorithm not in {"logistic", "xgboost", "auto"}:
        raise ValueError("algorithm must be logistic, xgboost or auto")
    if not 0 < review_fraction < 1:
        raise ValueError("review_fraction must be in (0,1)")
    df = make_transactions(n, seed).sort_values(TIME).reset_index(drop=True)
    validate_data(df)
    first, second = int(0.6 * n), int(0.8 * n)
    train_df, valid_df, test_df = df.iloc[:first], df.iloc[first:second], df.iloc[second:]
    if train_df[TARGET].nunique() != 2 or valid_df[TARGET].nunique() != 2:
        raise ValueError("Insufficient labels in train or validation partition")
    candidates = []
    experiments = []
    for kind, params in _candidate_models(seed, algorithm):
        # Fit each supervised/anomaly component once, mix scores on validation.
        base = HybridFraudModel(seed=seed, classifier_kind=kind, classifier_params=params).fit(
            train_df[FEATURES], train_df[TARGET]
        )
        components = base.predict_components(valid_df[FEATURES])
        for weight in (0.0, 0.2, 0.4):
            scores = (1 - weight) * components["supervised_score"].to_numpy() + (
                weight * components["anomaly_score"].to_numpy()
            )
            threshold = threshold_for_review_capacity(scores, review_fraction)
            metrics = evaluate(valid_df[TARGET], scores, threshold)
            description = {"algorithm": kind, "params": params, "anomaly_weight": weight,
                           "validation_pr_auc": metrics["pr_auc"], "threshold": threshold}
            experiments.append(description)
            candidates.append((metrics["pr_auc"], -weight, base, weight, threshold, metrics, description))
    _, _, model, weight, threshold, valid_metrics, choice = max(
        candidates, key=lambda result: (result[0], result[1])
    )
    model.anomaly_weight = weight
    test_score = model.predict_components(test_df[FEATURES])["risk_score"].to_numpy()
    test_metrics = evaluate(test_df[TARGET], test_score, threshold)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "threshold": threshold, "features": FEATURES,
                "model_version": "0.2.0", "review_fraction": review_fraction}
    joblib.dump(artifact, out / "model.joblib")
    result = {"dataset": "synthetic", "n": n, "seed": seed,
              "split": {"train": len(train_df), "validation": len(valid_df), "test": len(test_df)},
              "fraud_rates": {"train": float(train_df[TARGET].mean()),
                              "validation": float(valid_df[TARGET].mean()),
                              "test": float(test_df[TARGET].mean())},
              "selected": choice, "validation": valid_metrics, "test": test_metrics,
              "experiments": experiments}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if register_model or os.getenv("MLFLOW_TRACKING_URI"):
        from .tracking import log_run
        log_run(model, choice, test_metrics, out, seed, n, register_model)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--review-fraction", type=float, default=0.05)
    parser.add_argument("--algorithm", choices=["logistic", "xgboost", "auto"], default="logistic")
    parser.add_argument("--register-model", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.n, args.seed, args.output_dir, args.review_fraction,
                           args.algorithm, args.register_model), indent=2))


if __name__ == "__main__":
    main()
