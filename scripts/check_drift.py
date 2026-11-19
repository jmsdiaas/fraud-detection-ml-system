"""Run a reproducible drift report against two synthetic temporal windows."""
import argparse
import json
from pathlib import Path

from fraudml.data import make_transactions
from fraudml.monitor import drift_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/drift-report.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    df = make_transactions(4000, seed=42)
    report = drift_report(df.iloc[:2000], df.iloc[2000:], threshold=args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
