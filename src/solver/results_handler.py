# src/solver/results_handler.py

import os
import json
import numpy as np

def save_simulation_metadata(simulation_instance, output_dir):
    """
    Saves simulation configuration and mesh definition to JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)

    config_filepath = os.path.join(output_dir, "config.json")
    mesh_filepath = os.path.join(output_dir, "mesh.json")
    readme_filepath = os.path.join(output_dir, "readme.txt")

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
            # Convert numpy arrays in bc_info to lists if they exist
            if 'cell_indices' in serializable_bc_info and isinstance(serializable_bc_info['cell_indices'], np.ndarray):
                serializable_bc_info['cell_indices'] = serializable_bc_info['cell_indices'].tolist()
            if 'face_indices' in serializable_bc_info and isinstance(serializable_bc_info['face_indices'], np.ndarray):
                serializable_bc_info['face_indices'] = serializable_bc_info['face_indices'].tolist()
            
            mesh_data["boundary_conditions"][bc_name] = serializable_bc_info

    with open(mesh_filepath, 'w') as f:
        json.dump(mesh_data, f, indent=2)
    print(f"Saved mesh definition to: {mesh_filepath}")

    # Create a simple README file
    with open(readme_filepath, 'w') as f:
        f.write("Fluid Dynamics Simulation Results\n")
        f.write("---------------------------------\n")
        f.write(f"Input file: {os.path.basename(simulation_instance.input_file_path)}\n")
        f.write(f"Simulation started: {simulation_instance.start_time}\n")
        f.write(f"Total simulation time: {simulation_instance.total_time} s\n")
        f.write(f"Time step: {simulation_instance.time_step} s\n")
        f.write(f"Grid dimensions: {simulation_instance.grid_shape[0]}x{simulation_instance.grid_shape[1]}x{simulation_instance.grid_shape[2]}\n")
        f.write("\nField snapshots are saved in the 'fields' subdirectory.")
        f.write("\nFinal summary is in 'final_summary.json'.")
    print(f"Created readme: {readme_filepath}")


def save_field_snapshot(step_count, velocity_field, pressure_field, fields_dir):
    """
    Saves the current velocity and pressure fields to a JSON file.
    """
    os.makedirs(fields_dir, exist_ok=True)
    
    # Format step_count to have leading zeros for consistent file naming
    filename = f"step_{step_count:04d}.json"
    filepath = os.path.join(fields_dir, filename)

    # Convert numpy arrays to lists for JSON serialization
    # Use tolist() for the main arrays
    # Ensure nested arrays (like velocity_field[i][j][k] being [u,v,w]) are handled
    field_data = {
        "step": step_count,
        "velocity": velocity_field.tolist(), # Convert entire numpy array to nested list
        "pressure": pressure_field.tolist()  # Convert entire numpy array to nested list
    }

    with open(filepath, 'w') as f:
        json.dump(field_data, f, indent=2)
    print(f"Saved snapshot step {step_count} → {filepath}")


def save_final_summary(simulation_instance, output_dir):
    """
    Saves a summary of the simulation results, including a list of time points
    and history of a selected metric (e.g., average velocity magnitude).
    This function aggregates data that was saved incrementally.
    """
    final_summary_filepath = os.path.join(output_dir, "final_summary.json")
    fields_dir = os.path.join(output_dir, "fields")

    # --- FIX: Generate time_points based on actual simulation parameters ---
    # The time points should correspond to 0.01, 0.02, etc., up to total_time.
    # The range should start from time_step and go up to total_time, inclusive,
    # with a small tolerance for floating point accuracy.
    num_steps = int(round(simulation_instance.total_time / simulation_instance.time_step))
    
    # time_points should include the initial state (0.0) if relevant for plotting,
    # and then all subsequent steps. Given your snapshot saving starts at step 0,
    # and then increments, we should reflect that.
    # Let's generate points from 0 to total_time at intervals of time_step.
    # Adjust np.arange to correctly include the final point.
    time_points = np.linspace(0.0, simulation_instance.total_time, num_steps + 1).tolist()
    # If the first point (0.0) is not desired, remove it:
    # time_points = np.linspace(simulation_instance.time_step, simulation_instance.total_time, num_steps).tolist()

    velocity_history = []
    pressure_history = [] # Assuming you might want to save pressure history too

    # Load all saved field snapshots to compile history
    for step_idx in range(num_steps + 1): # Include step 0
        filename = f"step_{step_idx:04d}.json"
        filepath = os.path.join(fields_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                snapshot_data = json.load(f)
            
            # Extract relevant data, e.g., velocity field for history.
            # For simplicity, we'll just store the first cell's velocity as an example.
            # You might want to calculate an average or other metric.
            # If velocity_field is already a list of lists from .tolist(), direct append works.
            # If it's the full array, you'd save a summary metric.
            
            # For now, let's just append the velocity data from each step.
            # WARNING: Saving full velocity field history can create very large files.
            # Consider saving a derived metric (e.g., average velocity magnitude) instead.
            velocity_history.append(snapshot_data["velocity"])
            pressure_history.append(snapshot_data["pressure"]) # Example: append pressure

    summary_data = {
        "simulation_parameters": simulation_instance.input_data.get("simulation_parameters"),
        "time_points": time_points,
        "velocity_history": velocity_history,
        "pressure_history": pressure_history # Include pressure history
        # Add other summary metrics as needed (e.g., average pressure, total kinetic energy)
    }

    with open(final_summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=4) # Use indent=4 for better readability of large JSONs
    print(f"Saved final summary to: {final_summary_filepath}")
    print("✅ Main Navier-Stokes simulation executed successfully.")

