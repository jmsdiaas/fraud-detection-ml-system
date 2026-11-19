"""Generate explanations on a synthetic sample."""
import argparse
import json
from pathlib import Path
import joblib
from fraudml.data import make_transactions
from fraudml.explain import explain_supervised
from fraudml.schema import FEATURES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="artifacts/model.joblib")
    p.add_argument("--output", default="artifacts/sample-explanations.json")
    args = p.parse_args()
    payload = joblib.load(args.model_path)  # trusted artifact only
    example = make_transactions(500, seed=17).iloc[:5]
    report = explain_supervised(payload["model"], example[FEATURES])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"method": report["method"], "rows": 5, "output": str(path)}, indent=2))

if __name__ == "__main__":
    main()
