import json
from fraudml.train import train


def test_train_writes_artifacts(tmp_path):
    metrics = train(n=3000, seed=10, output_dir=str(tmp_path))
    assert (tmp_path / "model.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["dataset"] == "synthetic"
    assert metrics["split"] == {"train": 1800, "validation": 600, "test": 600}
    assert metrics["test"]["pr_auc"] >= 0
