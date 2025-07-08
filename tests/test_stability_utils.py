# tests/test_stability_utils.py

import pytest
import numpy as np

from src.stability_utils import (
    check_field_validity,
    test_velocity_bounds,
    test_shape_match,
    run_stability_checks,
    compute_volatility  # If added during phase 1
)

def test_check_field_validity_with_nan_and_inf():
    field = np.array([[1.0, np.nan], [np.inf, -5.0]])
    result = check_field_validity(field, label="CorruptedField")
    assert result is False

def test_check_field_validity_with_valid_values():
    field = np.array([[1.0, 2.0], [-3.0, 4.0]])
    result = check_field_validity(field, label="CleanField")
    assert result is True

def test_velocity_bounds_exceeding_limit():
    velocity_field = np.ones((4, 4, 4, 3)) * 1e3
    result = test_velocity_bounds(velocity_field, velocity_limit=100.0)
    assert result is False

def test_velocity_bounds_within_limit():
    velocity_field = np.ones((4, 4, 4, 3)) * 50.0
    result = test_velocity_bounds(velocity_field, velocity_limit=100.0)
    assert result is True

def test_shape_match_success():
    a = np.zeros((4, 4, 4))
    b = np.zeros((4, 4, 4))
    assert test_shape_match(a, b, "A", "B") is True

def test_shape_match_failure():
    a = np.zeros((4, 4, 4))
    b = np.zeros((5, 4, 4))
    assert test_shape_match(a, b, "A", "B") is False

def test_run_stability_checks_spike_triggered():
    # Prepare a synthetic divergence field with large values
    divergence_field = np.ones((6, 6, 6)) * 1000.0
    divergence_field = np.pad(divergence_field, pad_width=1, mode="constant", constant_values=0.0)
    velocity_field = np.ones((8, 8, 8, 3)) * 10.0
    pressure_field = np.ones((8, 8, 8))
    pass_flag, metrics = run_stability_checks(
        velocity_field=velocity_field,
        pressure_field=pressure_field,
        divergence_field=divergence_field,
        step=5,
        max_allowed_divergence=0.5,
        velocity_limit=100.0,
        spike_factor=100.0
    )
    assert metrics["spike_triggered"] is True
    assert pass_flag is True  # Because divergence_mode defaults to "log"

def test_compute_volatility_returns_correct_delta_and_slope():
    delta, slope = compute_volatility(current_value=250.0, previous_value=100.0, step=5)
    assert delta == 150.0
    assert slope == 30.0

@pytest.mark.parametrize("current, previous, step, expected_slope", [
    (200.0, 100.0, 10, 10.0),
    (300.0, 250.0, 5, 10.0),
    (150.0, 150.0, 3, 0.0)
])
def test_volatility_parameterized(current, previous, step, expected_slope):
    _, slope = compute_volatility(current, previous, step)
    assert pytest.approx(slope) == expected_slope



