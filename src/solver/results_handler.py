# src/solver/results_handler.py

import json
import os
import numpy as np

def save_simulation_metadata(sim_instance, output_dir):
    """
    Saves static simulation metadata (config and mesh info) at the start.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    config_path = os.path.join(output_dir, 'config.json')
    mesh_path = os.path.join(output_dir, 'mesh.json')
    
    # Save config metadata
    config_data = {
        'total_time': sim_instance.total_time,
        'time_step': sim_instance.time_step,
        'grid_shape': sim_instance.grid_shape,
        'fluid_density': sim_instance.rho,
        'fluid_viscosity': sim_instance.nu
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved simulation config to: {config_path}")

    # Save mesh data
    boundary_conditions_data = {}
    for bc_name, bc_data in sim_instance.mesh_info['boundary_conditions'].items():
        # Safely get the BC value, checking for 'value' first, then 'pressure'
        bc_value = bc_data.get('value', bc_data.get('pressure', None))
        
        boundary_conditions_data[bc_name] = {
            'type': bc_data['type'],
            # Convert cell_indices to a list for JSON serialization
            'cell_indices': bc_data['cell_indices'].tolist() if isinstance(bc_data.get('cell_indices'), np.ndarray) else bc_data.get('cell_indices'),
            'value': bc_value
        }

    mesh_data = {
        'grid_shape': sim_instance.grid_shape,
        'dx': sim_instance.mesh_info['dx'],
        'dy': sim_instance.mesh_info['dy'],
        'dz': sim_instance.mesh_info['dz'],
        'boundary_conditions': boundary_conditions_data
    }

    with open(mesh_path, 'w') as f:
        json.dump(mesh_data, f, indent=2)
    print(f"Saved mesh data to: {mesh_path}")

def save_field_snapshot(step_number, velocity_field, pressure_field, fields_dir):
    """
    Saves the velocity and pressure fields for a specific time step.
    """
    # Create the fields subdirectory if it doesn't exist
    os.makedirs(fields_dir, exist_ok=True)
    
    # Format the filename with zero-padding (e.g., step_0000.json, step_0001.json)
    filename = f"step_{step_number:04d}.json"
    filepath = os.path.join(fields_dir, filename)
    
    # Convert NumPy arrays to lists for JSON serialization
    snapshot_data = {
        "velocity": velocity_field.tolist(),
        "pressure": pressure_field.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(snapshot_data, f, indent=2)
    print(f"DEBUG: Saved snapshot for step {step_number} to {filepath}")

def save_final_summary(sim_instance, output_dir):
    """
    Saves a final summary of the simulation run.
    """
    summary_path = os.path.join(output_dir, 'final_summary.json')
    summary_data = {
        'final_time': sim_instance.current_time,
        'time_steps': sim_instance.step_count
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved final summary to: {summary_path}")