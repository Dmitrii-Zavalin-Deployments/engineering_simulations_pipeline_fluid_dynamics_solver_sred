# tests/conftest.py

import pytest
import numpy as np

@pytest.fixture
def clean_velocity_field():
    """
    Synthetic velocity field with safe, uniform magnitudes.
    Shape: (4, 4, 4, 3)
    """
    return np.ones((4, 4, 4, 3)) * 10.0

@pytest.fixture
def spike_velocity_field():
    """
    Velocity field with a single spike to test overflow detection.
    """
    field = np.ones((5, 5, 5, 3)) * 8.0
    field[2, 2, 2] = np.array([1e5, -1e5, 1e5])
    return field

@pytest.fixture
def corrupted_field_with_nan():
    """
    2D scalar field with NaN to simulate corruption.
    """
    field = np.zeros((4, 4))
    field[0, 1] = np.nan
    return field

@pytest.fixture
def corrupted_field_with_inf():
    """
    2D scalar field with Inf to simulate corruption.
    """
    field = np.ones((4, 4))
    field[1, 2] = np.inf
    return field

@pytest.fixture
def default_reflex_config():
    """
    A basic configuration dictionary for AdaptiveScheduler tests.
    """
    return {
        "damping_enabled": True,
        "damping_factor": 0.1,
        "abort_divergence_threshold": 1e4,
        "abort_velocity_threshold": 1e3,
        "abort_cfl_threshold": 100.0,
        "projection_passes_max": 4,
        "divergence_spike_factor": 100.0,
        "max_consecutive_failures": 3
    }



