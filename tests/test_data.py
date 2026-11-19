import pytest
from fraudml.data import make_transactions, validate_data


def test_generation_is_deterministic():
    assert make_transactions(400, 5).equals(make_transactions(400, 5))


def test_fraud_is_binary_and_time_ordered():
    df = make_transactions(2000)
    validate_data(df)
    assert df.event_time.is_monotonic_increasing
    assert set(df.is_fraud.unique()) == {0, 1}


def test_invalid_data_rejected():
    df = make_transactions(400)
    df.loc[0, "amount"] = -1
    with pytest.raises(ValueError):
        validate_data(df)
