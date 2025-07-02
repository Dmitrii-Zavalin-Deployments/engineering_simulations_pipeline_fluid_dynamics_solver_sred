import json
import numpy as np
import sys
import os
from preprocessing.identify_boundary_nodes import identify_boundary_nodes

def pre_process_input(input_file_path: str, output_file_path: str):
    """
    Loads a JSON input file, preprocesses it, and saves the processed data.
    The main preprocessing step is to identify boundary nodes for each boundary condition.
    
    Args:
        input_file_path (str): Path to the raw JSON input file.
        output_file_path (str): Path to save the processed JSON output file.
    """
    print(f"Loading raw input file from: {input_file_path}")
    try:
        with open(input_file_path, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {input_file_path}. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Pre-processing input data...")

    # --- 1. Validate and extract domain definition ---
    domain = input_data.get("domain_definition")
    if not domain:
        raise ValueError("Missing 'domain_definition' in input file.")
        
    nx = domain.get("nx")
    ny = domain.get("ny")
    nz = domain.get("nz")
    
    if not all([nx, ny, nz]):
        raise ValueError("Grid dimensions (nx, ny, nz) must be specified.")
    
    dx = (domain["max_x"] - domain["min_x"]) / nx
    dy = (domain["max_y"] - domain["min_y"]) / ny
    dz = (domain["max_z"] - domain["min_z"]) / nz

    # --- 2. Initialize mesh_info dictionary ---
    mesh_info = {
        "grid_shape": [nx, ny, nz],
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "min_x": domain["min_x"],
        "min_y": domain["min_y"],
        "min_z": domain["min_z"],
        "max_x": domain["max_x"],
        "max_y": domain["max_y"],
        "max_z": domain["max_z"]
    }

    # --- 3. Process Boundary Conditions ---
    raw_bcs = input_data.get("boundary_conditions", {})
    processed_bcs = {}

    if isinstance(raw_bcs, dict):
        print("Detected dictionary-based boundary conditions.")
        for name, bc_data in raw_bcs.items():
            processed_bcs[name] = bc_data
    elif isinstance(raw_bcs, list):
        print("Detected list-based boundary conditions.")
        for i, bc_data in enumerate(raw_bcs):
            label = bc_data.get("label", f"bc_{i}")
            processed_bcs[label] = bc_data
    else:
        raise ValueError("Boundary conditions must be a dictionary or a list of objects.")

    mesh_info["boundary_conditions"] = processed_bcs

    # --- 4. Identify boundary nodes and map them to grid indices ---
    mesh_faces = input_data.get("mesh", {}).get("boundary_faces")
    identify_boundary_nodes(mesh_info, mesh_faces)

    # --- 5. Build final output dictionary ---
    processed_data = {
        "mesh_info": mesh_info,
        "fluid_properties": input_data.get("fluid_properties", {}),
        "initial_conditions": input_data.get("initial_conditions", {}),
        "simulation_parameters": input_data.get("simulation_parameters", {}),
        "mesh": {
            "boundary_faces": mesh_faces
        }
    }

    # --- 6. Save processed JSON ---
    print(f"Saving pre-processed data to: {output_file_path}")
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print("✅ Pre-processing successful.")
    except IOError as e:
        print(f"Error: Could not save output file to {output_file_path}. Details: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pre_process_input.py <input_file_path> <output_file_path>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    pre_process_input(input_path, output_path)



