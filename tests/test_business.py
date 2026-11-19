import numpy as np
import pytest
from fraudml.business import Costs, ScoreCalibrator, optimize_policy, total_cost

def test_calibration_bounds():
    score = np.linspace(0, 1, 80)
    labels = np.array([0] * 70 + [1] * 10)
    c = ScoreCalibrator().fit(score, labels)
    assert np.all((c.predict_proba(score) > 0) & (c.predict_proba(score) < 1))

def test_policy_budget_and_cost():
    y = np.array([0] * 90 + [1] * 10)
    amount = np.repeat(100.0, 100)
    scores = np.linspace(0, 1, 100)
    result = optimize_policy(y, amount, scores, Costs(), .10)
    assert result["review_rate"] <= .10 + 1e-10
    assert total_cost(y, amount, scores, result["threshold"], Costs())["total_cost"] <= 1000

def test_invalid_capacity():
    with pytest.raises(ValueError):
        optimize_policy([0, 1], [10, 10], [.1, .9], Costs(), 0)
