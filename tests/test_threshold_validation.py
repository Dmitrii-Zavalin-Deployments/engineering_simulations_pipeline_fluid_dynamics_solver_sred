# tests/test_threshold_validation.py

import json
import pytest
import os

@pytest.fixture(scope="module")
def thresholds():
    path = os.path.join("src", "test_thresholds.json")
    with open(path) as f:
        return json.load(f)

def test_top_level_sections_exist(thresholds):
    required_keys = [
        "divergence_tests",
        "velocity_tests",
        "residual_tests",
        "pressure_tests",
        "projection_effectiveness",
        "cfl_tests",
        "test_behavior",
        "volatility_tests"
    ]
    for key in required_keys:
        assert key in thresholds, f"Missing section: {key}"

def test_divergence_threshold_values(thresholds):
    div = thresholds["divergence_tests"]
    assert div["max_divergence_threshold"] > 0
    assert div["spike_factor"] >= 10.0

def test_velocity_thresholds_reasonable(thresholds):
    vtest = thresholds["velocity_tests"]
    assert vtest["velocity_magnitude_max"] > vtest["velocity_magnitude_warning"]
    assert vtest["nan_inf_allowed"] is False

def test_projection_effectiveness_values(thresholds):
    proj = thresholds["projection_effectiveness"]
    assert 0.0 <= proj["minimum_reduction_percent"] <= 100.0
    assert 0.0 <= proj["failure_tolerance_percent"] <= 100.0

def test_volatility_threshold_values(thresholds):
    vtest = thresholds["volatility_tests"]
    assert vtest["warning_threshold"] < vtest["max_slope_per_step"]
    assert vtest["max_slope_per_step"] < 1e4
    assert vtest["warning_threshold"] >= 0.0



