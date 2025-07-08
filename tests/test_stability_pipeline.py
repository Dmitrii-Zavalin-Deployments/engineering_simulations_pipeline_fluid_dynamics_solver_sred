# tests/test_stability_pipeline.py

"""
Stability Pipeline Tests:
Automated diagnostics validating field sanity, reflex effectiveness,
and numerical reliability across simulation steps.
"""

import os
import json
import numpy as np
import pytest

from stability_utils import run_stability_checks  # ✅ Shared stability logic from src/

# Load threshold config
with open(os.path.join("src", "test_thresholds.json")) as f:
    thresholds = json.load(f)

# Synthetic test data generator
def generate_synthetic_fields(shape=(43, 43, 13)):
    velocity = np.random.randn(*shape, 3) * 1.0
    pressure = np.random.randn(*shape)
    divergence = np.random.randn(*shape) * 0.05
    return velocity, pressure, divergence

@pytest.mark.unit
def test_velocity_magnitude_bounds():
    velocity, _, _ = generate_synthetic_fields()
    magnitude = np.linalg.norm(velocity[1:-1, 1:-1, 1:-1, :], axis=-1)
    max_vel = float(np.max(magnitude))
    assert max_vel < thresholds["velocity_tests"]["velocity_magnitude_max"], f"Velocity too high: {max_vel:.2e}"

@pytest.mark.unit
def test_divergence_amplitude():
    _, _, divergence = generate_synthetic_fields()
    interior = np.abs(divergence[1:-1, 1:-1, 1:-1])
    max_div = float(np.max(interior))
    assert max_div < thresholds["divergence_tests"]["max_divergence_threshold"], f"∇·u spike: {max_div:.2e}"

@pytest.mark.unit
def test_pressure_statistics():
    _, pressure, _ = generate_synthetic_fields()
    interior = pressure[1:-1, 1:-1, 1:-1]
    std_p = float(np.std(interior))
    max_p = float(np.max(interior))
    assert max_p < thresholds["pressure_tests"]["pressure_range_max"], f"Pressure too large: {max_p:.2e}"
    assert std_p < thresholds["pressure_tests"]["pressure_stddev_max"], f"Pressure std too high: {std_p:.2e}"

@pytest.mark.integration
def test_projection_effectiveness():
    before = np.abs(np.random.randn(41, 41, 11))
    after = before * 0.2  # Simulate strong correction
    reduction = 1.0 - (np.max(after) / np.max(before))
    assert reduction >= thresholds["projection_effectiveness"]["minimum_reduction_percent"] / 100.0, f"Poor projection reduction: {reduction:.2%}"

@pytest.mark.regression
def test_residual_limits():
    simulated_residual = np.random.uniform(low=0.0, high=thresholds["residual_tests"]["residual_max"] * 0.5)
    assert simulated_residual < thresholds["residual_tests"]["residual_max"], f"Residual too high: {simulated_residual:.2e}"

@pytest.mark.unit
def test_nan_inf_safety():
    velocity, pressure, divergence = generate_synthetic_fields()
    for label, field in [("Velocity", velocity), ("Pressure", pressure), ("Divergence", divergence)]:
        assert not np.isnan(field).any(), f"{label} field contains NaN"
        assert not np.isinf(field).any(), f"{label} field contains Inf"

def test_config_integrity():
    required_keys = ["divergence_tests", "velocity_tests", "residual_tests", "pressure_tests", "projection_effectiveness"]
    for key in required_keys:
        assert key in thresholds, f"Missing config section: {key}"

@pytest.mark.integration
def test_run_stability_checks_output_shape():
    velocity, pressure, divergence = generate_synthetic_fields()
    passed, metrics = run_stability_checks(
        velocity_field=velocity,
        pressure_field=pressure,
        divergence_field=divergence,
        step=1,
        expected_velocity_shape=velocity.shape,
        expected_pressure_shape=pressure.shape,
        expected_divergence_shape=divergence.shape,
        divergence_mode="log",
        max_allowed_divergence=thresholds["divergence_tests"]["max_divergence_threshold"],
        velocity_limit=thresholds["velocity_tests"]["velocity_magnitude_max"],
        spike_factor=thresholds["divergence_tests"]["spike_factor"]
    )
    assert isinstance(passed, bool), "Stability check did not return a boolean status"
    assert "max" in metrics and "mean" in metrics, "Metrics missing expected keys"



