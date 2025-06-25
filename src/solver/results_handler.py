# src/solver/results_handler.py

import json
import os
import sys

def save_simulation_results(simulation_instance, output_file_path):
    """
    Saves the final velocity and pressure fields to a JSON file.
    Takes the simulation instance as input to access final state data.
    """
    print(f"Saving results to: {output_file_path}")
    try:
        # Convert NumPy arrays to lists for JSON serialization
        results = {
            'final_time': simulation_instance.total_time,
            'time_steps': int(simulation_instance.total_time / simulation_instance.time_step),
            'grid_shape': simulation_instance.grid_shape,
            'dx': simulation_instance.dx,
            'dy': simulation_instance.dy,
            'dz': simulation_instance.dz,
            'final_velocity_u': simulation_instance.u.tolist(),
            'final_velocity_v': simulation_instance.v.tolist(),
            'final_velocity_w': simulation_instance.w.tolist(),
            'final_pressure': simulation_instance.p.tolist()
        }
        
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print("Results saved successfully.")
        
    except Exception as e:
        print(f"Error saving results: {e}", file=sys.stderr)
        sys.exit(1)