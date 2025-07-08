# tests/stability_tests.py

import numpy as np

_previous_divergence_max = None  # Global tracker for divergence spike analysis


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
    Checks for excessive divergence and tracks instability metrics.

    Returns:
        (bool, dict): stabilitypass flag, divergence trend diagnostics
    """
    global _previous_divergence_max

    interior = divergence_field[1:-1, 1:-1, 1:-1]
    interior = np.nan_to_num(interior, nan=0.0, posinf=0.0, neginf=0.0)

    max_div = float(np.max(np.abs(interior)))
    mean_div = float(np.mean(np.abs(interior)))

    print(f"📏 ∇·u check: max={max_div:.4e}, mean={mean_div:.4e}")

    metrics = {
        "max": max_div,
        "mean": mean_div,
        "delta": None,
        "slope": None,
        "spike_triggered": False
    }

    if _previous_divergence_max is not None:
        delta = max_div - _previous_divergence_max
        slope = delta / max(1, step)
        metrics["delta"] = delta
        metrics["slope"] = slope
        print(f"📊 ∇·u trend: Δ={delta:.4e}, Δ/step={slope:.4e}")

    _previous_divergence_max = max_div

    if max_div > spike_factor * max_allowed_divergence:
        print(f"🚨 ∇·u spike: {max_div:.4e} > {spike_factor:.1f}×{max_allowed_divergence:.4e}")
        metrics["spike_triggered"] = True
        return mode != "strict", metrics

    if max_div > max_allowed_divergence:
        print(f"⚠️ ∇·u above threshold: {max_div:.4e} > {max_allowed_divergence:.4e}")
        return mode != "strict", metrics

    return True, metrics


def test_velocity_bounds(velocity_field: np.ndarray, velocity_limit: float = 10.0):
    """
    Validates that velocity magnitude stays within specified bounds.
    """
    interior = velocity_field[1:-1, 1:-1, 1:-1, :]
    magnitude = np.linalg.norm(np.nan_to_num(interior), axis=-1)
    max_vel = float(np.max(magnitude))
    print(f"⚡ Velocity magnitude: max={max_vel:.4e}")
    return max_vel < velocity_limit


def test_shape_match(field_a: np.ndarray, field_b: np.ndarray, label_a="A", label_b="B"):
    """
    Checks if two fields have the same shape.
    """
    match = field_a.shape == field_b.shape
    print(f"📐 Shape match: {label_a}={field_a.shape}, {label_b}={field_b.shape}, match={match}")
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
    Runs stability diagnostics and trend tracking for a given simulation step.

    Returns:
        (bool, dict): status flag and divergence metrics
    """
    print(f"\n🧪 Stability Checks @ Step {step}")
   pass_flag = True

   pass_flag &= check_field_validity(velocity_field, "Velocity")
   pass_flag &= check_field_validity(pressure_field, "Pressure")
   pass_flag &= check_field_validity(divergence_field, "Divergence")

    div_check, div_metrics = test_divergence_stability(
        divergence_field,
        max_allowed_divergence,
        divergence_mode,
        step,
        spike_factor
    )
   pass_flag &= div_check

   pass_flag &= test_velocity_bounds(velocity_field, velocity_limit)

    if expected_velocity_shape:
       pass_flag &= test_shape_match(velocity_field, np.zeros(expected_velocity_shape), "Velocity", "Expected")
    if expected_pressure_shape:
       pass_flag &= test_shape_match(pressure_field, np.zeros(expected_pressure_shape), "Pressure", "Expected")
    if expected_divergence_shape:
       pass_flag &= test_shape_match(divergence_field, np.zeros(expected_divergence_shape), "Divergence", "Expected")

    if notpass_flag:
        print(f"❌ Step {step}: Stability FAILED. Diagnostic review advised.")
    else:
        print(f"✅ Step {step}: StabilitypassED.")

    returnpass_flag, div_metrics



