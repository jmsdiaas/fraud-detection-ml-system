"""Post-hoc calibration and held-out financial evaluation, synthetic data only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from fraudml.business import Costs, ScoreCalibrator, total_cost, optimize_policy, calibration_metrics, costs_dict
from fraudml.data import make_transactions
from fraudml.schema import FEATURES, TARGET, TIME

def evaluate(n: int = 12000, seed: int = 42, model_path: str = "artifacts/model.joblib",
             output_path: str = "artifacts/business-evaluation.json", costs: Costs | None = None,
             capacity: float = .05) -> dict:
    costs = costs or Costs()
    payload = joblib.load(model_path)  # Only use trusted locally generated artifacts
    df = make_transactions(n, seed).sort_values(TIME).reset_index(drop=True)
    validation, test = df.iloc[int(.6 * n):int(.8 * n)], df.iloc[int(.8 * n):]
    calibration, policy = np.array_split(validation, 2)
    raw_cal = payload["model"].predict_components(calibration[FEATURES])["risk_score"].to_numpy()
    raw_policy = payload["model"].predict_components(policy[FEATURES])["risk_score"].to_numpy()
    raw_test = payload["model"].predict_components(test[FEATURES])["risk_score"].to_numpy()
    calibrator = ScoreCalibrator().fit(raw_cal, calibration[TARGET])
    policy_score = calibrator.predict_proba(raw_policy)
    test_score = calibrator.predict_proba(raw_test)
    decision = optimize_policy(policy[TARGET], policy["amount"], policy_score, costs, capacity)
    holdout = total_cost(test[TARGET], test["amount"], test_score, decision["threshold"], costs)
    no_review = total_cost(test[TARGET], test["amount"], test_score, float("inf"), costs)
    report = {"dataset": "synthetic", "model_version": payload.get("model_version", "unknown"),
              "calibration_method": "platt_logistic", "policy_selection": "validation_second_half",
              "test_policy_frozen": True, "cost_assumptions": costs_dict(costs),
              "max_review_fraction": capacity, "policy_validation": decision,
              "test_cost": holdout, "test_no_review_cost": no_review["total_cost"],
              "test_avoided_cost_vs_no_review": no_review["total_cost"] - holdout["total_cost"],
              "calibration_test": calibration_metrics(test[TARGET], raw_test, test_score)}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default="artifacts/model.joblib")
    p.add_argument("--output-path", default="artifacts/business-evaluation.json")
    p.add_argument("--capacity", type=float, default=.05)
    p.add_argument("--review-cost", type=float, default=2.0)
    p.add_argument("--friction-cost", type=float, default=5.0)
    p.add_argument("--prevented-fraction", type=float, default=.85)
    args = p.parse_args()
    costs = Costs(review_cost=args.review_cost, false_positive_friction=args.friction_cost,
                  prevented_fraction=args.prevented_fraction)
    print(json.dumps(evaluate(args.n, args.seed, args.model_path, args.output_path, costs, args.capacity), indent=2))

if __name__ == "__main__":
    main()
