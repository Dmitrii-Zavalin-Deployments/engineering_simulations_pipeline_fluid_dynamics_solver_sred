# src/stability_utils.py

import numpy as np
import warnings
import sys

def check_field_validity(field, label="field"):
    """
    Returns False if the field contains NaN or Inf; True otherwise.
    Useful for early detection of numerical corruption.
    """
    if np.isnan(field).any() or np.isinf(field).any():
        print(f"[WARNING] {label} contains NaN or Inf values.")
        return False
    return True

def velocity_bounds_check(field, velocity_limit):
    """
    Returns True if all velocity magnitudes are below the threshold.
    Assumes velocity field shape: (nx, ny, nz, 3)
    """
    magnitudes = np.linalg.norm(field, axis=-1)
    max_magnitude = np.max(magnitudes)
    if max_magnitude >= velocity_limit:
        print(f"[WARNING] Velocity magnitude exceeds limit: {max_magnitude:.2f} > {velocity_limit}")
        return False
    return True

def compute_volatility(current_value, previous_value, step):
    """
    Computes volatility metrics between time slices.
    Returns delta and slope (rate of change per step).
    """
    delta = current_value - previous_value
    slope = delta / max(step, 1)
    return delta, slope

def get_threshold(thresh_dict, key, default, silent=False):
    import sys
    sys.stderr.write(f"[DEBUG] Looking for '{key}' in: {thresh_dict}\n")
    val = thresh_dict.get(key, default)
    if val == default and not silent:
        warnings.warn(f"[THRESHOLD FALLBACK] Key '{key}' not found. Using default: {default}")
    return val



