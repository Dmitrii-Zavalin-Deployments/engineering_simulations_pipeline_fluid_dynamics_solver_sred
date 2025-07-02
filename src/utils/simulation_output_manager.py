# src/utils/simulation_output_manager.py

import os
from datetime import datetime
from utils.io_utils import save_json, convert_numpy_to_list

def setup_simulation_output_directory(simulation_instance, output_dir):
    """
    Sets up the output directory structure and saves initial simulation metadata.
    This includes config.json and mesh.json.
    Creation of readme.txt has been removed as per user request.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fields"), exist_ok=True)

    config_filepath = os.path.join(output_dir, "config.json")
    mesh_filepath = os.path.join(output_dir, "mesh.json")

    # Prepare config snapshot (excluding large fields)
    config_data = {
        "domain_definition": simulation_instance.input_data.get("domain_definition"),
        "fluid_properties": simulation_instance.input_data.get("fluid_properties"),
        "initial_conditions": simulation_instance.input_data.get("initial_conditions"),
        "simulation_parameters": simulation_instance.input_data.get("simulation_parameters"),
    }
    save_json(config_data, config_filepath)
    print(f"Saved simulation config to: {config_filepath}")

    # Prepare mesh metadata and sanitize NumPy arrays
    mesh_data = {
        "grid_shape": simulation_instance.mesh_info.get("grid_shape"),
        "dx": simulation_instance.mesh_info.get("dx"),
        "dy": simulation_instance.mesh_info.get("dy"),
        "dz": simulation_instance.mesh_info.get("dz"),
        "cell_coords": simulation_instance.mesh_info.get("cell_coords"),
        "face_coords": simulation_instance.mesh_info.get("face_coords"),
        "boundary_conditions": simulation_instance.mesh_info.get("boundary_conditions", {})
    }

    # ✅ Clean all remaining NumPy arrays before writing
    cleaned_mesh_data = convert_numpy_to_list(mesh_data)

    save_json(cleaned_mesh_data, mesh_filepath)
    print(f"Saved mesh definition to: {mesh_filepath}")



