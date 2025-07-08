# src/stability_utils.py

import numpy as np

_previous_divergence_max = None  # Tracker for divergence trend

def check_field_validity(field: np.ndarray, label: str = "field"):
    """
    Checks for NaNs, Infs, and reports min/max with overflow-safe diagnostics.
    """
    try:
        with np.errstate(over='raise', invalid='raise'):
            has_nan = np.isnan(field).any()
            has_inf = np.isinf(field).any()
            min_val = np.nanmin(field)
            max_val = np.nanmax(field)
        print(f"🔍 {label}: min={min_val:.4e}, max={max_val:.4e}, has_nan={has_nan}, has_inf={has_inf}")
        return not (has_nan or has_inf)
    except FloatingPointError as e:
        print(f"❌ {label} field caused overflow: {e}")
        return False

def test_velocity_bounds(velocity_field: np.ndarray, velocity_limit: float):
    """
    Validates magnitude of velocity field is physically bounded.
    """
    try:
        interior = velocity_field[1:-1, 1:-1, 1:-1, :]
        with np.errstate(over='raise', invalid='raise'):
            magnitude = np.linalg.norm(np.nan_to_num(interior), axis=-1)
            max_vel = float(np.max(magnitude))
        print(f"⚡ Velocity magnitude: max={max_vel:.4e}")
        return max_vel < velocity_limit
    except FloatingPointError as e:
        print(f"❌ Velocity bounds check failed due to overflow: {e}")
        return False

def test_shape_match(field_a: np.ndarray, field_b: np.ndarray, label_a="A", label_b="B"):
    """
    Verifies shape consistency between fields.
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
    Composite diagnostic logic to assess simulation field stability.
    Returns:
        (bool, dict): Pass status and divergence metrics
    """
    global _previous_divergence_max

    print(f"\n🧪 Stability Checks @ Step {step}")
    pass_flag = True

    pass_flag &= check_field_validity(velocity_field, "Velocity")
    pass_flag &= check_field_validity(pressure_field, "Pressure")
    pass_flag &= check_field_validity(divergence_field, "Divergence")

    try:
        interior = divergence_field[1:-1, 1:-1, 1:-1]
        interior = np.nan_to_num(interior, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over='raise', invalid='raise'):
            max_div = float(np.max(np.abs(interior)))
            mean_div = float(np.mean(np.abs(interior)))
        print(f"📏 ∇·u check: max={max_div:.4e}, mean={mean_div:.4e}")
    except FloatingPointError as e:
        print(f"❌ Divergence calculation failed due to overflow: {e}")
        return False, {"max": np.inf, "mean": np.inf, "spike_triggered": True}

    metrics = {
        "max": max_div,
        "mean": mean_div,
        "delta": None,
        "slope": None,
        "spike_triggered": False
    }

    if _previous_divergence_max is not None:
        try:
            with np.errstate(over='raise', invalid='raise'):
                delta = max_div - _previous_divergence_max
                slope = delta / max(1, step)
            metrics["delta"] = delta
            metrics["slope"] = slope
            print(f"📊 ∇·u trend: Δ={delta:.4e}, Δ/step={slope:.4e}")
        except FloatingPointError as e:
            print(f"⚠️ Divergence trend computation failed: {e}")

    _previous_divergence_max = max_div

    try:
        with np.errstate(over='raise', invalid='raise'):
            if max_div > spike_factor * max_allowed_divergence:
                print(f"🚨 ∇·u spike: {max_div:.4e} > {spike_factor:.1f}×{max_allowed_divergence:.4e}")
                metrics["spike_triggered"] = True
                pass_flag = divergence_mode != "strict"
            elif max_div > max_allowed_divergence:
                print(f"⚠️ ∇·u above threshold: {max_div:.4e} > {max_allowed_divergence:.4e}")
                pass_flag = divergence_mode != "strict"
    except FloatingPointError as e:
        print(f"⚠️ Threshold comparison failed: {e}")
        pass_flag = False

    pass_flag &= test_velocity_bounds(velocity_field, velocity_limit)

    if expected_velocity_shape:
        pass_flag &= test_shape_match(velocity_field, np.zeros(expected_velocity_shape), "Velocity", "Expected")
    if expected_pressure_shape:
        pass_flag &= test_shape_match(pressure_field, np.zeros(expected_pressure_shape), "Pressure", "Expected")
    if expected_divergence_shape:
        pass_flag &= test_shape_match(divergence_field, np.zeros(expected_divergence_shape), "Divergence", "Expected")

    if not pass_flag:
        print(f"❌ Step {step}: Stability FAILED.")
    else:
        print(f"✅ Step {step}: Stability PASSED.")

    return pass_flag, metrics



