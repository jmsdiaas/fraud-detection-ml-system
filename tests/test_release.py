import hashlib
import joblib
import pandas as pd
import pytest
from fraudml.release import build_release
from fraudml.schema import FEATURES

def test_release_with_independent_windows(tmp_path):
    report = build_release(3000, output_dir=str(tmp_path))
    assert list(report["split_sizes"].values()) == [1650, 450, 300, 300, 300]
    data = (tmp_path / "model.joblib").read_bytes()
    assert hashlib.sha256(data).hexdigest() == report["model_sha256"]
    bundle = joblib.load(tmp_path / "model.joblib")
    assert not bundle["approved_for_real_transactions"]
    raw = bundle["model"].predict_components(pd.DataFrame([{f: 1 for f in FEATURES}])).risk_score.to_numpy()
    assert 0 <= bundle["calibrator"].predict_proba(raw)[0] <= 1

def test_release_rejects_small_samples(tmp_path):
    with pytest.raises(ValueError):
        build_release(100, output_dir=str(tmp_path))
