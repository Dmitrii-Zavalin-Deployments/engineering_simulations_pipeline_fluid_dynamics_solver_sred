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
        encoding = data.get("mask_encoding", {"fluid": 1})
        fluid_code = encoding.get("fluid", 1)
        return [v == fluid_code for v in raw]
    else:
        # Fallback: simple boolean conversion
        return [bool(v) for v in data]

def decode_geometry_mask(config):
    """
    Decodes the geometry mask using mask_encoding from input config.
    
    Parameters:
    - config (dict): Parsed JSON input configuration
    
    Returns:
    - List[bool]: Decoded fluid mask
    """
    raw = config["geometry_definition"]["geometry_mask_flat"]
    fluid_code = config["geometry_definition"]["mask_encoding"]["fluid"]
    return [v == fluid_code for v in raw]

def get_grid_centers(domain):
    """
    Computes all grid center coordinates for a structured domain.

    Parameters:
    - domain (dict): domain_definition block from config

    Returns:
    - List[Tuple[float, float, float]]: Grid cell center coordinates
    """
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    x_centers = [domain["min_x"] + (i + 0.5) * dx for i in range(domain["nx"])]
    y_centers = [domain["min_y"] + (j + 0.5) * dy for j in range(domain["ny"])]
    z_centers = [domain["min_z"] + (k + 0.5) * dz for k in range(domain["nz"])]

    return [(x, y, z) for x in x_centers for y in y_centers for z in z_centers]



