# src/solver/results_handler.py

import os
import json
import numpy as np

# save_simulation_metadata has been moved to src/utils/simulation_output_manager.py
# save_final_summary has been removed as per user request.

def save_field_snapshot(step_count, velocity_field, pressure_field, fields_dir):
    """
    Saves the current velocity and pressure fields to a JSON file.
    """
    os.makedirs(fields_dir, exist_ok=True)
    
    # Format step_count to have leading zeros for consistent file naming
    filename = f"step_{step_count:04d}.json"
    filepath = os.path.join(fields_dir, filename)

    # Convert numpy arrays to lists for JSON serialization
    field_data = {
        "step": step_count,
        "velocity": velocity_field.tolist(), # Convert entire numpy array to nested list
        "pressure": pressure_field.tolist()  # Convert entire numpy array to nested list
    }

    with open(filepath, 'w') as f:
        json.dump(field_data, f, indent=2)
    print(f"Saved snapshot step {step_count} → {filepath}")

# The save_final_summary function has been removed as per your request
# def save_final_summary(simulation_instance, output_dir):
#     """
#     Saves a summary of the simulation results, including a list of time points
#     and history of a selected metric (e.g., average velocity magnitude).
#     This function aggregates data that was saved incrementally.
#     """
#     final_summary_filepath = os.path.join(output_dir, "final_summary.json")
#     fields_dir = os.path.join(output_dir, "fields")

#     num_steps = int(round(simulation_instance.total_time / simulation_instance.time_step))
    
#     time_points = np.linspace(0.0, simulation_instance.total_time, num_steps + 1).tolist()

#     velocity_history = []
#     pressure_history = [] 

#     for step_idx in range(num_steps + 1): 
#         filename = f"step_{step_idx:04d}.json"
#         filepath = os.path.join(fields_dir, filename)
#         if os.path.exists(filepath):
#             with open(filepath, 'r') as f:
#                 snapshot_data = json.load(f)
            
#             velocity_history.append(snapshot_data["velocity"])
#             pressure_history.append(snapshot_data["pressure"])

#     summary_data = {
#         "simulation_parameters": simulation_instance.input_data.get("simulation_parameters"),
#         "time_points": time_points,
#         "velocity_history": velocity_history,
#         "pressure_history": pressure_history
#     }

#     with open(final_summary_filepath, 'w') as f:
#         json.dump(summary_data, f, indent=4)
#     print(f"Saved final summary to: {final_summary_filepath}")
#     print("✅ Main Navier-Stokes simulation executed successfully.")