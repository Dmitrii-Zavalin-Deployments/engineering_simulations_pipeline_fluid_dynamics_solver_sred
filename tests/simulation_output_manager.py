# src/simulation_output_manager.py

import os
import json
import numpy as np

def convert_numpy_to_list(arr):
    """
    Converts a NumPy array to a nested Python list for JSON serialization.
    """
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr

def write_divergence_snapshot(path, step, divergence_field, velocity_field, cfl_value, overflow=True, mode="log"):
    """
    Centralized snapshot writer for divergence overflow logging.
    
    Args:
        path (str): File path to write the snapshot.
        step (int): Simulation step index.
        divergence_field (np.ndarray): Divergence values (3D).
        velocity_field (np.ndarray): Velocity vector field (4D).
        cfl_value (float): CFL number at this step.
        overflow (bool): Whether overflow was detected.
        mode (str): Logging mode for divergence recording.
    """
    max_div = np.max(divergence_field)
    max_vel = np.max(np.linalg.norm(velocity_field, axis=-1))
    shape = divergence_field.shape

    snapshot = {
        "step": step,
        "max_divergence": float(max_div),
        "max_velocity": float(max_vel),
        "global_cfl": float(cfl_value),
        "overflow_detected": bool(overflow),
        "divergence_mode": mode,
        "field_shape": list(shape),
        "divergence_values": convert_numpy_to_list(divergence_field)
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)



