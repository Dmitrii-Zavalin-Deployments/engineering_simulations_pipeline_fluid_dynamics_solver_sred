# tests/test_stability_utils.py

import pytest
import numpy as np

from src.stability_utils import (
    check_field_validity,
    velocity_bounds_check,
    compute_volatility
)

# --- Field Validity Tests ---

def test_check_field_validity_with_nan_and_inf():
    field = np.array([[1.0, np.nan], [np.inf, -5.0]])
    result = check_field_validity(field, label="CorruptedField")
    assert result is False

def test_check_field_validity_with_valid_values():
    field = np.array([[1.0, 2.0], [-3.0, 4.0]])
    result = check_field_validity(field, label="CleanField")
    assert result is True

# --- Velocity Bounds Tests ---

def test_velocity_bounds_exceeding_limit():
    velocity_field = np.ones((4, 4, 4, 3)) * 1000.0
    result = velocity_bounds_check(velocity_field, velocity_limit=100.0)
    assert result is False

def test_velocity_bounds_within_limit():
    velocity_field = np.ones((4, 4, 4, 3)) * 50.0
    result = velocity_bounds_check(velocity_field, velocity_limit=100.0)
    assert result is True

# --- Volatility Metric Tests ---

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

# --- Threshold Sensitivity Tests (Synthetic Behavior) ---

@pytest.mark.parametrize("current, previous, step, threshold, expected_trigger", [
    (51.0, 0.0, 1, 50.0, True),      # Exceeds threshold
    (49.5, 0.0, 1, 50.0, False),     # Just below threshold
    (50.0, 0.0, 1, 50.0, False),     # Equal to threshold
    (75.0, 50.0, 5, 5.0, False)      # Slope = 5.0, not exceeding threshold
])
def test_volatility_warning_trigger(current, previous, step, threshold, expected_trigger):
    _, slope = compute_volatility(current, previous, step)
    triggered = slope > threshold
    assert triggered == expected_trigger



