"""Explicit MLflow experiment tracking and optional registered model version."""
from __future__ import annotations

import os
from pathlib import Path


def log_run(model, selection: dict, test_metrics: dict, output_dir: Path,
            seed: int, n: int, register_model: bool = False) -> str:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise ImportError("Install MLflow: pip install -e '.[tracking]'") from exc
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(output_dir / 'mlflow.db').resolve()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-synthetic-v2"))
    with mlflow.start_run(run_name=f"{selection['algorithm']}-hybrid") as run:
        mlflow.set_tags({"dataset": "synthetic", "model_stage": "experimental",
                         "approval_status": "unvalidated_for_production"})
        mlflow.log_params({"seed": seed, "dataset_size": n, "algorithm": selection["algorithm"],
                           "anomaly_weight": selection["anomaly_weight"],
                           **{f"classifier_{k}": v for k, v in selection["params"].items()}})
        mlflow.log_metrics({"validation_pr_auc": selection["validation_pr_auc"],
                            **{f"test_{k}": v for k, v in test_metrics.items()
                               if isinstance(v, (int, float)) and v is not None}})
        mlflow.log_artifact(str(output_dir / "metrics.json"))
        # Registration is opt-in: immutable version but no automatic stage promotion.
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            registered_model_name=os.getenv("MLFLOW_MODEL_NAME", "fraud-hybrid-synthetic")
            if register_model else None,
        )
        return run.info.run_id
