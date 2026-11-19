"""Lightweight XGBoost integration and model search tests."""
import pytest

from fraudml.data import make_transactions
from fraudml.model import HybridFraudModel
from fraudml.schema import FEATURES, TARGET
from fraudml.train import train


def test_xgboost_can_fit_and_score():
    pytest.importorskip("xgboost")
    df = make_transactions(1500)
    model = HybridFraudModel(classifier_kind="xgboost",
                            classifier_params={"n_estimators": 15}).fit(df[FEATURES], df[TARGET])
    scores = model.predict_components(df.iloc[:4][FEATURES])
    assert scores["risk_score"].between(0, 1).all()


def test_xgboost_training_selection(tmp_path):
    pytest.importorskip("xgboost")
    result = train(n=1500, output_dir=str(tmp_path), algorithm="xgboost")
    assert result["selected"]["algorithm"] == "xgboost"
    assert (tmp_path / "model.joblib").exists()


def test_invalid_algorithm(tmp_path):
    with pytest.raises(ValueError):
        train(n=1500, output_dir=str(tmp_path), algorithm="invalid")
