"""Time-aware train/validation/test; model selection never touches test labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
from .data import make_transactions, validate_data
from .model import HybridFraudModel
from .evaluation import evaluate, threshold_for_review_capacity
from .schema import FEATURES, TARGET, TIME


def train(n: int = 12000, seed: int = 42, output_dir: str = "artifacts", review_fraction: float = .05) -> dict:
    df = make_transactions(n, seed).sort_values(TIME).reset_index(drop=True)
    validate_data(df)
    i, j = int(.6 * n), int(.8 * n)
    tr, va, te = df.iloc[:i], df.iloc[i:j], df.iloc[j:]
    if tr[TARGET].nunique() != 2:
        raise ValueError("Insufficient class variety in training period; try a larger n")
    candidate_results = []
    for weight in [0.0, 0.2, 0.4]:
        model = HybridFraudModel(anomaly_weight=weight, seed=seed).fit(tr[FEATURES], tr[TARGET])
        val_scores = model.predict_components(va[FEATURES])["risk_score"].to_numpy()
        threshold = threshold_for_review_capacity(val_scores, review_fraction)
        metrics = evaluate(va[TARGET], val_scores, threshold)
        candidate_results.append((metrics["pr_auc"], -weight, model, threshold, metrics))
    _, _, best, threshold, valid_metrics = max(candidate_results, key=lambda a: (a[0], a[1]))
    test_scores = best.predict_components(te[FEATURES])["risk_score"].to_numpy()
    test_metrics = evaluate(te[TARGET], test_scores, threshold)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": best, "threshold": threshold, "features": FEATURES,
               "model_version": "0.1.0", "review_fraction": review_fraction}
    joblib.dump(payload, out / "model.joblib")
    result = {"dataset": "synthetic", "n": n, "seed": seed,
              "split": {"train": len(tr), "validation": len(va), "test": len(te)},
              "fraud_rates": {"train": float(tr[TARGET].mean()), "validation": float(va[TARGET].mean()), "test": float(te[TARGET].mean())},
              "anomaly_weight": best.anomaly_weight, "validation": valid_metrics, "test": test_metrics}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        import mlflow
    except ImportError:
        pass
    else:
        mlflow.set_tracking_uri("file:" + str(out.resolve() / "mlruns"))
        with mlflow.start_run(run_name="synthetic_hybrid_fraud"):
            mlflow.log_params({"seed": seed, "n": n, "anomaly_weight": best.anomaly_weight,
                              "review_fraction": review_fraction})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float)) and v is not None})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--review-fraction", type=float, default=.05)
    args = parser.parse_args()
    print(json.dumps(train(args.n, args.seed, args.output_dir, args.review_fraction), indent=2))


if __name__ == "__main__":
    main()
