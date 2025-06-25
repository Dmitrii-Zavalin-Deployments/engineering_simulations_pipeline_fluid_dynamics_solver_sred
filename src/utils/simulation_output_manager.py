# src/utils/simulation_output_manager.py

import os
import json
import numpy as np
from datetime import datetime

def setup_simulation_output_directory(simulation_instance, output_dir):
    """
    Sets up the output directory structure and saves initial simulation metadata.
    This includes config.json and mesh.json.
    Creation of readme.txt has been removed as per user request.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fields"), exist_ok=True) # Ensure fields directory exists

    config_filepath = os.path.join(output_dir, "config.json")
    mesh_filepath = os.path.join(output_dir, "mesh.json")
    # readme_filepath and its creation block removed as per user request

    # Save simulation parameters (excluding large arrays like fields)
    config_data = {
        "domain_definition": simulation_instance.input_data.get("domain_definition"),
        "fluid_properties": simulation_instance.input_data.get("fluid_properties"),
        "initial_conditions": simulation_instance.input_data.get("initial_conditions"),
        "simulation_parameters": simulation_instance.input_data.get("simulation_parameters"),
    }
    with open(config_filepath, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved simulation config to: {config_filepath}")

    # Save mesh definition (including boundary faces if present)
    mesh_data = {
        "grid_shape": simulation_instance.mesh_info.get("grid_shape"),
        "dx": simulation_instance.mesh_info.get("dx"),
        "dy": simulation_instance.mesh_info.get("dy"),
        "dz": simulation_instance.mesh_info.get("dz"),
        "cell_coords": simulation_instance.mesh_info.get("cell_coords").tolist() if simulation_instance.mesh_info.get("cell_coords") is not None else None,
        "face_coords": simulation_instance.mesh_info.get("face_coords").tolist() if simulation_instance.mesh_info.get("face_coords") is not None else None,
        "boundary_conditions": {}
    }
    
    # Safely handle boundary_conditions for JSON serialization
    if 'boundary_conditions' in simulation_instance.mesh_info:
        for bc_name, bc_info in simulation_instance.mesh_info['boundary_conditions'].items():
            serializable_bc_info = bc_info.copy()
            if 'cell_indices' in serializable_bc_info and isinstance(serializable_bc_info['cell_indices'], np.ndarray):
                serializable_bc_info['cell_indices'] = serializable_bc_info['cell_indices'].tolist()
            if 'face_indices' in serializable_bc_info and isinstance(serializable_bc_info['face_indices'], np.ndarray):
                serializable_bc_info['face_indices'] = serializable_bc_info['face_indices'].tolist()
            
            mesh_data["boundary_conditions"][bc_name] = serializable_bc_info

    with open(mesh_filepath, 'w') as f:
        json.dump(mesh_data, f, indent=2)
    print(f"Saved mesh definition to: {mesh_filepath}")