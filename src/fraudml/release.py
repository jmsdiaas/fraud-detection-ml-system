"""Time-separated, synthetic-only candidate release pipeline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
from sklearn.metrics import average_precision_score
from .business import Costs, ScoreCalibrator, calibration_metrics, optimize_policy, total_cost
from .data import make_transactions, validate_data
from .evaluation import evaluate
from .model import HybridFraudModel
from .schema import FEATURES, TARGET, TIME

VERSION = "0.4.0"

def build_release(n: int = 12000, seed: int = 42, output_dir: str = "artifacts/release",
                  algorithm: str = "logistic", capacity: float = .05) -> dict:
    if algorithm not in ("logistic", "xgboost", "auto"):
        raise ValueError("Unsupported algorithm")
    if n < 3000 or not 0 < capacity <= 1:
        raise ValueError("Need at least 3000 events and valid review capacity")
    data = make_transactions(n, seed).sort_values(TIME).reset_index(drop=True)
    validate_data(data)
    cuts = [0, int(.55*n), int(.70*n), int(.80*n), int(.90*n), n]
    fit, selection, calibration, policy, test = [data.iloc[cuts[i]:cuts[i+1]] for i in range(5)]
    for name, part in (("fit", fit), ("selection", selection), ("calibration", calibration),
                       ("policy", policy), ("test", test)):
        if part[TARGET].nunique() != 2:
            raise ValueError(f"{name} needs both labels; increase sample size")
    options = []
    kinds = ("logistic", "xgboost") if algorithm == "auto" else (algorithm,)
    for kind in kinds:
        model = HybridFraudModel(seed=seed, classifier_kind=kind).fit(fit[FEATURES], fit[TARGET])
        parts = model.predict_components(selection[FEATURES])
        for weight in (0., .2, .4):
            scores = (1-weight)*parts.supervised_score.to_numpy() + weight*parts.anomaly_score.to_numpy()
            ap = float(average_precision_score(selection[TARGET], scores))
            options.append((ap, -weight, kind, weight, model))
    ap, _, kind, weight, chosen = max(options, key=lambda row: (row[0], row[1]))
    chosen.anomaly_weight = weight

    def risk(frame):
        return chosen.predict_components(frame[FEATURES]).risk_score.to_numpy()

    calibrator = ScoreCalibrator().fit(risk(calibration), calibration[TARGET])
    costs = Costs()
    decision = optimize_policy(policy[TARGET], policy["amount"],
                               calibrator.predict_proba(risk(policy)), costs, capacity)
    raw_test = risk(test)
    scored_test = calibrator.predict_proba(raw_test)
    threshold = decision["threshold"]
    metrics = evaluate(test[TARGET], scored_test, threshold)
    finance = total_cost(test[TARGET], test["amount"], scored_test, threshold, costs)
    control = total_cost(test[TARGET], test["amount"], scored_test, float("inf"), costs)
    diagnostics = calibration_metrics(test[TARGET], raw_test, scored_test)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": chosen, "calibrator": calibrator, "threshold": threshold,
               "features": FEATURES, "model_version": VERSION, "policy": "review",
               "dataset": "synthetic", "risk_probability_is_calibrated": True,
               "approved_for_real_transactions": False}
    binary = out / "model.joblib"
    joblib.dump(payload, binary)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    report = {"version": VERSION, "dataset": "synthetic", "seed": seed, "n": n,
              "split_sizes": {"fit": len(fit), "selection": len(selection),
                              "calibration": len(calibration), "policy": len(policy), "test": len(test)},
              "selected_algorithm": kind, "selection_pr_auc": ap, "anomaly_weight": weight,
              "capacity": capacity, "policy_validation": decision, "test_metrics": metrics,
              "test_calibration": diagnostics, "test_cost": finance,
              "test_no_review_cost": control["total_cost"], "model_sha256": digest,
              "limitations": "Synthetic only; never approved for production payment decisions"}
    (out / "release.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/release")
    parser.add_argument("--algorithm", choices=["logistic", "xgboost", "auto"], default="logistic")
    parser.add_argument("--capacity", type=float, default=.05)
    args = parser.parse_args()
    print(json.dumps(build_release(args.n, args.seed, args.output_dir, args.algorithm, args.capacity), indent=2))

if __name__ == "__main__":
    main()
