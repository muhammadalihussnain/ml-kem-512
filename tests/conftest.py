"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def ml_kem_512_params():
    """ML-KEM-512 parameters."""
    return {
        "n": 256,
        "k": 2,
        "q": 3329,
        "eta1": 3,
        "eta2": 2,
        "du": 10,
        "dv": 4,
    }


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def test_polynomial(ml_kem_512_params):
    """Generate a test polynomial."""
    np.random.seed(42)
    return np.random.randint(0, ml_kem_512_params["q"], size=ml_kem_512_params["n"])
