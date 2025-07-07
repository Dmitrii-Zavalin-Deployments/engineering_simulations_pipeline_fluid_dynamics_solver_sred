# tests/stability_tests.py

import numpy as np
import sys

_previous_divergence_max = None  # Module-level cache for divergence tracking

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


def test_divergence_stability(
    divergence_field: np.ndarray,
    max_allowed_divergence: float = 3e-2,
    mode: str = "log",
    step: int = 0,
    spike_factor: float = 100.0
):
    """
    Checks if divergence magnitude exceeds threshold and detects growth spikes.

    Modes:
        - "strict": fails the test if above threshold
        - "log": logs warning but returns True
        - "none": disables check entirely
    """
    global _previous_divergence_max

    if mode == "none":
        print(f"📏 ∇·u check skipped (mode='none')")
        return True

    interior = divergence_field[1:-1, 1:-1, 1:-1]
    interior = np.nan_to_num(interior, nan=0.0, posinf=0.0, neginf=0.0)
    max_div = np.max(np.abs(interior))
    mean_div = np.mean(np.abs(interior))

    print(f"📏 ∇·u check: max={max_div:.4e}, mean={mean_div:.4e}")

    if _previous_divergence_max is not None:
        delta = max_div - _previous_divergence_max
        rate = delta / max(1, step)
        print(f"📊 ∇·u change from previous: Δ={delta:.4e}, Δ/step={rate:.4e}")
        if delta > 0 and mode == "strict":
            print("📈 ∇·u is increasing — consider inspecting projection or reducing dt.")

    _previous_divergence_max = max_div

    if max_div > spike_factor * max_allowed_divergence:
        print(f"🚨 ∇·u spike detected: max={max_div:.4e} > {spike_factor:.1f}×{max_allowed_divergence:.4e}")
        return mode != "strict"

    if max_div > max_allowed_divergence:
        print(f"⚠️ ∇·u exceeds allowed threshold: max={max_div:.4e} > {max_allowed_divergence:.4e}")
        return mode != "strict"

    return True


def test_velocity_bounds(velocity_field: np.ndarray, velocity_limit: float = 10.0):
    """
    Ensures velocity magnitudes stay within physical bounds.
    """
    interior_v = velocity_field[1:-1, 1:-1, 1:-1, :]
    velocity_mag = np.linalg.norm(np.nan_to_num(interior_v), axis=-1)

    max_vel = np.max(velocity_mag)
    print(f"⚡ Velocity magnitude: max={max_vel:.4e}")

    return max_vel < velocity_limit


def test_shape_match(field_a: np.ndarray, field_b: np.ndarray, label_a="A", label_b="B"):
    """
    Ensures two fields have matching shapes.
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
    expected_divergence_shape: tuple = None,
    divergence_mode: str = "log",
    max_allowed_divergence: float = 3e-2,
    velocity_limit: float = 10.0,
    spike_factor: float = 100.0
):
    """
    Runs all stability diagnostics for the current simulation step.
    Optionally verifies that fields match expected shapes and padding.

    Returns:
        bool: True if all checks pass, False if any fail.
    """
    print(f"\n🧪 Stability Checks @ Step {step}")
    tests_passed = True

    tests_passed &= check_field_validity(velocity_field, "Velocity")
    tests_passed &= check_field_validity(pressure_field, "Pressure")
    tests_passed &= check_field_validity(divergence_field, "Divergence")

    tests_passed &= test_divergence_stability(
        divergence_field,
        max_allowed_divergence=max_allowed_divergence,
        mode=divergence_mode,
        step=step,
        spike_factor=spike_factor
    )

    tests_passed &= test_velocity_bounds(velocity_field, velocity_limit=velocity_limit)

    if expected_velocity_shape:
        tests_passed &= test_shape_match(
            velocity_field,
            np.zeros(expected_velocity_shape),
            label_a="Velocity", label_b="Expected"
        )
    if expected_pressure_shape:
        tests_passed &= test_shape_match(
            pressure_field,
            np.zeros(expected_pressure_shape),
            label_a="Pressure", label_b="Expected"
        )
    if expected_divergence_shape:
        tests_passed &= test_shape_match(
            divergence_field,
            np.zeros(expected_divergence_shape),
            label_a="Divergence", label_b="Expected"
        )

    if not tests_passed:
        print(f"❌ Step {step}: Stability test FAILED. Halting or inspecting recommended.")
    else:
        print(f"✅ Step {step}: Stability test PASSED.")

    return tests_passed



