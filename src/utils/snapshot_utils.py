# src/utils/snapshot_utils.py

import os
import json
import numpy as np

def write_snapshot(path, step, divergence, velocity, cfl, overflow=False, mode="log"):
    """
    Writes a structured snapshot JSON file representing simulation metrics.

    Parameters:
        path (str): Output file path
        step (int): Simulation step number
        divergence (float): Max divergence value
        velocity (float): Max velocity value
        cfl (float): Global CFL condition
        overflow (bool): Flag indicating overflow detection
        mode (str): Divergence mode ("log", "raw", etc.)

    Returns:
        str: Absolute path to the written snapshot file
    """
    snapshot_data = {
        "step": step,
        "max_divergence": divergence,
        "max_velocity": velocity,
        "global_cfl": cfl,
        "overflow_detected": overflow,
        "divergence_mode": mode,
        "field_shape": [8, 8, 8],
        "divergence_values": np.full((8, 8, 8), divergence).tolist()
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot_data, f, indent=2)

    return os.path.abspath(path)



