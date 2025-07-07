# src/utils/simulation_output_manager.py

import os
from datetime import datetime
from utils.io_utils import save_json, convert_numpy_to_list
import numpy as np

def setup_simulation_output_directory(simulation_instance, output_dir):
    """
    Sets up output directories and saves initial simulation metadata.
    Includes sanitized config.json and mesh.json snapshots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fields"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    config_filepath = os.path.join(output_dir, "config.json")
    mesh_filepath = os.path.join(output_dir, "mesh.json")

    config_data = {
        "domain_definition": simulation_instance.input_data.get("domain_definition"),
        "fluid_properties": simulation_instance.input_data.get("fluid_properties"),
        "initial_conditions": simulation_instance.input_data.get("initial_conditions"),
        "simulation_parameters": simulation_instance.input_data.get("simulation_parameters"),
        "start_time": simulation_instance.start_time
    }
    save_json(config_data, config_filepath)
    print(f"📝 Saved simulation config to: {config_filepath}")

    mesh_data = {
        "grid_shape": simulation_instance.mesh_info.get("grid_shape"),
        "dx": simulation_instance.mesh_info.get("dx"),
        "dy": simulation_instance.mesh_info.get("dy"),
        "dz": simulation_instance.mesh_info.get("dz"),
        "cell_coords": simulation_instance.mesh_info.get("cell_coords"),
        "face_coords": simulation_instance.mesh_info.get("face_coords"),
        "boundary_conditions": simulation_instance.mesh_info.get("boundary_conditions", {})
    }

    cleaned_mesh_data = convert_numpy_to_list(mesh_data)
    save_json(cleaned_mesh_data, mesh_filepath)
    print(f"📐 Saved mesh definition to: {mesh_filepath}")


def log_divergence_snapshot(divergence_field, step_count, output_dir, additional_meta=None):
    """
    Logs divergence metrics, solver health, and runtime events per step.

    Args:
        divergence_field (np.ndarray): Full divergence field with ghost zones.
        step_count (int): Current simulation step.
        output_dir (str): Path to root simulation output folder.
        additional_meta (dict): Optional metadata (e.g., dt, residual, recovery_triggered, projection_passes, damping_applied, smoother_iterations)
    """
    interior = divergence_field[1:-1, 1:-1, 1:-1]
    interior = np.nan_to_num(interior, nan=0.0, posinf=0.0, neginf=0.0)

    stats = {
        "step": step_count,
        "timestamp": datetime.now().isoformat(),
        "max_divergence": float(np.max(np.abs(interior))),
        "mean_divergence": float(np.mean(np.abs(interior))),
        "min_divergence": float(np.min(interior)),
        "std_divergence": float(np.std(interior))
    }

    if additional_meta:
        for key, val in additional_meta.items():
            if isinstance(val, (np.float32, np.float64)):
                stats[key] = float(val)
            elif isinstance(val, (np.ndarray, list)):
                stats[key] = convert_numpy_to_list(val)
            else:
                stats[key] = val

    log_path = os.path.join(output_dir, "logs", f"divergence_step_{step_count:04d}.json")
    save_json(stats, log_path)
    print(f"📦 Logged divergence metrics → {log_path}")



