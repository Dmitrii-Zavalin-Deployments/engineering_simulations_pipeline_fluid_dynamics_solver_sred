# tests/conftest.py

import pytest
import numpy as np
import os
import json

SNAPSHOT_PATH = "data/testing-output-run/navier_stokes_output/divergence_snapshot.json"
THRESHOLD_PATH = os.path.join("src", "test_thresholds.json")
SCHEMA_PATH = os.path.join("schema", "thresholds.schema.json")

@pytest.fixture
def clean_velocity_field():
    """Synthetic velocity field with safe, uniform magnitudes. Shape: (4, 4, 4, 3)"""
    return np.ones((4, 4, 4, 3)) * 10.0

@pytest.fixture
def spike_velocity_field():
    """Velocity field with a single spike to test overflow detection."""
    field = np.ones((5, 5, 5, 3)) * 8.0
    field[2, 2, 2] = np.array([1e5, -1e5, 1e5])
    return field

@pytest.fixture
def corrupted_field_with_nan():
    """2D scalar field with NaN to simulate corruption."""
    field = np.zeros((4, 4))
    field[0, 1] = np.nan
    return field

@pytest.fixture
def corrupted_field_with_inf():
    """2D scalar field with Inf to simulate corruption."""
    field = np.ones((4, 4))
    field[1, 2] = np.inf
    return field

@pytest.fixture
def complete_reflex_config():
    """
    Full configuration dictionary for AdaptiveScheduler tests.
    Avoids fallback warnings by including all expected keys.
    """
    return {
        "damping_enabled": True,
        "damping_factor": 0.1,
        "divergence_spike_factor": 100.0,
        "abort_divergence_threshold": 1e6,
        "abort_velocity_threshold": 1e6,
        "abort_cfl_threshold": 1e6,
        "projection_passes_max": 4,
        "max_consecutive_failures": 3
    }

@pytest.fixture(autouse=True, scope="module")
def write_snapshot_for_tests():
    """
    Auto-generated snapshot file for overflow-related tests.
    Runs once per module.
    """
    snapshot_data = {
        "step": 42,
        "max_divergence": float("inf"),
        "max_velocity": 1e6,
        "global_cfl": 2.5,
        "overflow_detected": True,
        "divergence_mode": "log",
        "field_shape": [8, 8, 8],
        "divergence_values": np.full((8, 8, 8), 1e3).tolist()
    }
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot_data, f, indent=2)

@pytest.fixture(scope="module")
def loaded_thresholds():
    """Loads test_thresholds.json for use in threshold validation tests."""
    with open(THRESHOLD_PATH) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def threshold_schema():
    """Loads schema for validating structure of test thresholds."""
    if os.path.isfile(SCHEMA_PATH):
        with open(SCHEMA_PATH) as f:
            return json.load(f)
    return None



