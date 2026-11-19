import numpy as np
from fraudml.monitor import psi


def test_psi_identical_distributions():
    arr = np.arange(100, dtype=float)
    assert psi(arr, arr) == 0


def test_psi_detects_shift():
    arr = np.arange(100, dtype=float)
    assert psi(arr, arr + 75) > 0.1
