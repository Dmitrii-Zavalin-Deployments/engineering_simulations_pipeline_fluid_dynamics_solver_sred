# tests/stability_tests.py

import numpy as np
import sys

def check_field_validity(field: np.ndarray, label: str = "field"):
    """
    Checks for NaNs, Infs, and prints min/max for a given field.
    """
    has_nan = np.isnan(field).any()
    has_inf = np.isinf(field).any()
    min_val = np.nanmin(field)
    max_val = np.nanmax(field)

    print(f"🔍 {label}: min={min_val:.4e}, max={max_val:.4e}, has_nan={has_nan}, has_inf={has_inf}")
    return not (has_nan or has_inf)


def test_divergence_stability(divergence_field: np.ndarray, max_allowed_divergence: float = 1e-4):
    """
    Checks if divergence magnitude exceeds threshold.
    """
    interior = divergence_field[1:-1, 1:-1, 1:-1]
    max_div = np.max(np.abs(interior))
    mean_div = np.mean(np.abs(interior))

    print(f"📏 ∇·u check: max={max_div:.4e}, mean={mean_div:.4e}")
    return max_div <= max_allowed_divergence


def test_velocity_bounds(velocity_field: np.ndarray, velocity_limit: float = 10.0):
    """
    Ensures velocity magnitudes stay within physical bounds.
    """
    interior_v = velocity_field[1:-1, 1:-1, 1:-1, :]
    velocity_mag = np.linalg.norm(interior_v, axis=-1)

    max_vel = np.max(velocity_mag)
    print(f"⚡ Velocity magnitude: max={max_vel:.4e}")

    return max_vel < velocity_limit


def test_shape_match(field_a: np.ndarray, field_b: np.ndarray, label_a="A", label_b="B"):
    """
    Ensures two fields have matching shapes. Useful for prolongation safety.
    """
    match = field_a.shape == field_b.shape
    print(f"📐 Shape match check: {label_a} shape={field_a.shape}, {label_b} shape={field_b.shape}, match={match}")
    return match


def run_stability_checks(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    divergence_field: np.ndarray,
    step: int,
    expected_velocity_shape: tuple = None,
    expected_pressure_shape: tuple = None,
    expected_divergence_shape: tuple = None
):
    """
    Runs all stability diagnostics for the current simulation step.
    Optionally verifies that fields match expected shapes.
    """
    print(f"\n🧪 Stability Checks @ Step {step}")
    tests_passed = True

    tests_passed &= check_field_validity(velocity_field, "Velocity")
    tests_passed &= check_field_validity(pressure_field, "Pressure")
    tests_passed &= check_field_validity(divergence_field, "Divergence")

    tests_passed &= test_divergence_stability(divergence_field)
    tests_passed &= test_velocity_bounds(velocity_field)

    if expected_velocity_shape:
        tests_passed &= test_shape_match(velocity_field, np.zeros(expected_velocity_shape), label_a="Velocity", label_b="Expected")
    if expected_pressure_shape:
        tests_passed &= test_shape_match(pressure_field, np.zeros(expected_pressure_shape), label_a="Pressure", label_b="Expected")
    if expected_divergence_shape:
        tests_passed &= test_shape_match(divergence_field, np.zeros(expected_divergence_shape), label_a="Divergence", label_b="Expected")

    if not tests_passed:
        print(f"❌ Step {step}: Stability test FAILED. Halting or inspecting recommended.")
    else:
        print(f"✅ Step {step}: Stability test PASSED.")

    return tests_passed



