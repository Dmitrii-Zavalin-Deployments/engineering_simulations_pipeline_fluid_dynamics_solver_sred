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
    """
    Centralized accessor for threshold values.
    Logs a warning if a fallback value is used,
    unless silent=True (useful for test overrides or expected defaults).
    Also prints debug info to stderr for diagnostics.
    """
    sys.stderr.write(f"[DEBUG] Looking for '{key}' in: {thresh_dict}\n")
    val = thresh_dict.get(key, default)
    if val == default and not silent:
        warnings.warn(f"[THRESHOLD FALLBACK] Key '{key}' not found. Using default: {default}")
    return val

def validate_threshold_config(config, required_keys, silent=False):
    """
    Verifies that required threshold keys are present.
    Logs warnings for any missing keys unless silent=True.
    """
    missing = [key for key in required_keys if key not in config]
    if missing and not silent:
        warnings.warn(f"[CONFIG CHECK] Missing threshold keys: {missing}")
    return len(missing) == 0

class ReflexConfig:
    """
    Wrapper for threshold config access.
    Handles defaults, fallback detection, and structured retrieval.
    """
    def __init__(self, config_dict):
        self.config = config_dict

    def get(self, key, default=None, silent=False):
        return get_threshold(self.config, key, default, silent)

    def validate(self, required_keys):
        return validate_threshold_config(self.config, required_keys)



