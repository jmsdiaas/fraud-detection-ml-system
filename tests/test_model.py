import numpy as np
from fraudml.data import make_transactions
from fraudml.model import HybridFraudModel
from fraudml.schema import FEATURES, TARGET
from fraudml.evaluation import threshold_for_review_capacity


def test_hybrid_scores_are_bounded():
    df = make_transactions(2500)
    model = HybridFraudModel().fit(df[FEATURES], df[TARGET])
    scores = model.predict_components(df.iloc[:10][FEATURES])
    assert scores.shape == (10, 3)
    assert np.isfinite(scores.to_numpy()).all()
    assert ((scores >= 0) & (scores <= 1)).all().all()


def test_review_capacity():
    assert threshold_for_review_capacity(np.arange(100), .05) > 90
