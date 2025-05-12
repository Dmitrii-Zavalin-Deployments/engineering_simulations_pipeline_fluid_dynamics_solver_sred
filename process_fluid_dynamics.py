import json
import numpy as np

def load_boundary_conditions(file_path):
    """Load boundary conditions from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def process_fluid_dynamics(boundary_conditions):
    """Perform basic fluid dynamics calculations based on boundary conditions."""
    velocity_x = boundary_conditions.get("velocity_x", 1.0)
    velocity_y = boundary_conditions.get("velocity_y", 0.0)
    velocity_z = boundary_conditions.get("velocity_z", 0.0)
    
    flow_data = {
        "adjusted_velocity_x": velocity_x * np.random.uniform(0.9, 1.1),
        "adjusted_velocity_y": velocity_y * np.random.uniform(0.9, 1.1),
        "adjusted_velocity_z": velocity_z * np.random.uniform(0.9, 1.1),
    }
    
    return flow_data

def save_output(file_path, output_data):
    """Save the processed fluid dynamics output to JSON."""
    try:
        with open(file_path, 'w') as file:
            json.dump(output_data, file, indent=4)
        print(f"Output saved to {file_path}")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    input_path = "testing-input-output/boundary_conditions.json"
    output_path = "testing-input-output/fluid_flow_parameters.json"

    boundary_conditions = load_boundary_conditions(input_path)
    if boundary_conditions:
        flow_data = process_fluid_dynamics(boundary_conditions)
        save_output(output_path, flow_data)
    else:
        print("Failed to process fluid dynamics due to missing or invalid input.")



