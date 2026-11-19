from fraudml.data import make_transactions
from fraudml.explain import explain_supervised
from fraudml.model import HybridFraudModel
from fraudml.schema import FEATURES, TARGET

def test_logistic_explanation_is_additive():
    df = make_transactions(1200)
    model = HybridFraudModel().fit(df[FEATURES], df[TARGET])
    explanation = explain_supervised(model, df.iloc[:4][FEATURES])
    assert len(explanation["contributions"]) == 4
    assert len(explanation["contributions"][0]) == 9
