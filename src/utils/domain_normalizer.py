# src/utils/domain_normalizer.py

import logging

REQUIRED_KEYS = {
    "min_x": 0.0, "max_x": 1.0, "nx": 1,
    "min_y": 0.0, "max_y": 1.0, "ny": 1,
    "min_z": 0.0, "max_z": 1.0, "nz": 1
}

def normalize_domain(domain: dict) -> dict:
    """
    Ensures domain dictionary contains all required physical bounds and resolution keys.
    Fills in missing keys with safe defaults and logs a warning if normalization occurs.
    Also substitutes defaults for keys explicitly set to None.

    Args:
        domain (dict): Raw domain input (may be incomplete or contain None values)

    Returns:
        dict: Fully populated domain dictionary with defaults applied where necessary
    """
    normalized = domain.copy()
    missing_keys = []

    for key, default in REQUIRED_KEYS.items():
        if key not in normalized:
            normalized[key] = default
            missing_keys.append(key)
        elif normalized[key] is None:
            normalized[key] = default
            missing_keys.append(key)

    if missing_keys:
        logging.warning(
            f"⚠️ Domain normalization applied — missing keys substituted with defaults: {missing_keys}"
        )

    return normalized



