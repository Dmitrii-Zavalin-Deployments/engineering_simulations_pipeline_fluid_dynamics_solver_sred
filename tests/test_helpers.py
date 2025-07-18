# tests/test_helpers.py

import json
import os

def load_geometry_mask_bool(path: str):
    """
    Loads a geometry mask from a JSON file and returns a boolean array.
    This is a minimal fallback implementation intended for use in test environments.
    
    Parameters:
    - path (str): Path to the JSON mask file
    
    Returns:
    - List[bool]: Boolean mask array
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found → {path}")
    
    with open(path, "r") as f:
        data = json.load(f)

    # Defensive fallback: attempt to parse flat mask or boolean encoding
    if isinstance(data, dict) and "geometry_mask_flat" in data:
        raw = data["geometry_mask_flat"]
    else:
        raw = data

    # Convert to bool safely
    return [bool(v) for v in raw]



