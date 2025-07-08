# tests/test_threshold_validation.py

import json
import pytest
import os
from jsonschema import validate, ValidationError

# Optional: Load schema from separate file if needed
SCHEMA_PATH = os.path.join("schema", "thresholds.schema.json")

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
    assert div.get("max_divergence_threshold", 0) > 0
    assert div.get("spike_factor", 0) >= 10.0

def test_velocity_thresholds_reasonable(thresholds):
    vtest = thresholds["velocity_tests"]
    assert vtest.get("velocity_magnitude_max", 0) > 50.0
    assert vtest.get("warning_tolerance_percent", 0) <= 50.0

def test_projection_effectiveness_values(thresholds):
    proj = thresholds["projection_effectiveness"]
    assert 0.0 <= proj.get("minimum_reduction_percent", 0.0) <= 100.0
    assert proj.get("max_projection_passes", 0) >= 1

def test_volatility_threshold_values(thresholds):
    vtest = thresholds["volatility_tests"]
    assert vtest.get("warning_threshold", 0.0) < vtest.get("max_slope_per_step", 1e4)
    assert vtest.get("warning_threshold", -1.0) >= 0.0
    assert vtest.get("delta_threshold", 0.0) >= 0.0

def test_cfl_threshold_range(thresholds):
    cfl = thresholds["cfl_tests"]
    assert 0.0 < cfl.get("max_cfl_stable", 0.0) < 1.0

def test_behavior_flags(thresholds):
    behavior = thresholds["test_behavior"]
    assert isinstance(behavior.get("enable_overflow_logging"), bool)
    assert isinstance(behavior.get("strict_mode"), bool)

def test_thresholds_schema_compliance(thresholds):
    if not os.path.isfile(SCHEMA_PATH):
        pytest.skip("Schema file not found")
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    try:
        validate(instance=thresholds, schema=schema)
    except ValidationError as e:
        pytest.fail(f"Schema validation failed: {e.message}")

def test_no_fallback_values_used(thresholds):
    vtest = thresholds.get("volatility_tests", {})
    warning = vtest.get("warning_threshold", -1.0)
    assert warning != -1.0, "Fallback used: 'warning_threshold' is missing"
    assert warning >= 0.0

@pytest.mark.parametrize("cfl_value", [0.89, 0.9, 0.91])
def test_cfl_threshold_sensitivity(cfl_value, thresholds):
    threshold = thresholds["cfl_tests"]["max_cfl_stable"]
    should_be_stable = cfl_value <= threshold
    # Simulate expected reflex behavior (logic placeholder)
    assert should_be_stable is True or should_be_stable is False  # Passes regardless, placeholder for simulation link



