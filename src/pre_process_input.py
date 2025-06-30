# src/pre_process_input.py

import json
import sys
import os
import numpy as np

# --- REFINED FIX FOR ImportError ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END REFINED FIX ---

from preprocessing.identify_boundary_nodes import identify_boundary_nodes
from utils.io_utils import load_json_schema, convert_numpy_to_list, save_json
from utils.validation import validate_json_schema
from utils.mesh_utils import get_domain_extents, infer_uniform_grid_parameters

def pre_process_input_data(input_data):
    """
    Pre-processes raw input data from a JSON file into a structured format
    suitable for the fluid dynamics solver.
    """
    print("DEBUG (pre_process_input_data): Starting pre-processing.")

    # 1. Determine grid dimensions and domain extents
    boundary_faces = input_data["mesh"]["boundary_faces"]
    min_x, max_x, min_y, max_y, min_z, max_z = get_domain_extents(boundary_faces)

    all_x_coords, all_y_coords, all_z_coords = [], [], []
    for face in boundary_faces:
        for node_coords in face["nodes"].values():
            all_x_coords.append(node_coords[0])
            all_y_coords.append(node_coords[1])
            all_z_coords.append(node_coords[2])

    json_domain_def = input_data.get('domain_definition', {})
    json_nx = json_domain_def.get('nx')
    json_ny = json_domain_def.get('ny')
    json_nz = json_domain_def.get('nz')

    if json_nx is not None:
        nx = json_nx
        dx = (max_x - min_x) / nx
    else:
        dx, nx = infer_uniform_grid_parameters(min_x, max_x, all_x_coords, 'x', None)

    if json_ny is not None:
        ny = json_ny
        dy = (max_y - min_y) / ny
    else:
        dy, ny = infer_uniform_grid_parameters(min_y, max_y, all_y_coords, 'y', None)

    if json_nz is not None:
        nz = json_nz
        dz = (max_z - min_z) / nz
    else:
        dz, nz = infer_uniform_grid_parameters(min_z, max_z, all_z_coords, 'z', None)

    nx, ny, nz = max(1, nx), max(1, ny), max(1, nz)

    # Handle zero-extent dimensions for 2D or 1D problems
    TOLERANCE_ZERO_EXTENT = 1e-9
    if abs(max_x - min_x) < TOLERANCE_ZERO_EXTENT: dx, nx = 1.0, 1
    if abs(max_y - min_y) < TOLERANCE_ZERO_EXTENT: dy, ny = 1.0, 1
    if abs(max_z - min_z) < TOLERANCE_ZERO_EXTENT: dz, nz = 1.0, 1

    print(f"✔ Grid dimensions: nx={nx}, ny={ny}, nz={nz}")

    domain_settings = {
        "min_x": min_x, "max_x": max_x, "dx": dx, "nx": nx,
        "min_y": min_y, "max_y": max_y, "dy": dy, "ny": ny,
        "min_z": min_z, "max_z": max_z, "dz": dz, "nz": nz,
        "grid_shape": [nx, ny, nz]
    }

    # 2. Build the mesh_info dictionary
    mesh_info = {
        'grid_shape': [nx, ny, nz],
        'dx': dx, 'dy': dy, 'dz': dz,
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'min_z': min_z, 'max_z': max_z,
        'boundary_conditions': {} # Initialize an empty dict
    }

    # 3. Process and pass through boundary conditions, including pressure
    raw_boundary_conditions = input_data.get("boundary_conditions", [])
    if not isinstance(raw_boundary_conditions, list):
        raise ValueError("The 'boundary_conditions' field must be a list of condition objects.")

    for bc_config in raw_boundary_conditions:
        bc_label = bc_config.get("label")
        if not bc_label:
            print("Warning: Found a boundary condition without a 'label'. It will be skipped.")
            continue
            
        # Extract the fields to be applied (e.g., "velocity", "pressure")
        apply_to_fields = bc_config.get("apply_to", [])
        
        # Check that 'apply_to' is a list
        if not isinstance(apply_to_fields, list):
             raise ValueError(f"The 'apply_to' field for BC '{bc_label}' must be a list.")
             
        # Create a copy to pass through to mesh_info
        bc_data = bc_config.copy()
        
        # We need to map the fields requested in `apply_to`
        for field_name in apply_to_fields:
            if field_name in bc_data:
                # Store the value in mesh_info under the label
                # This ensures the preprocessor just passes the data through
                # without needing to know if it's velocity, pressure, etc.
                if bc_label not in mesh_info['boundary_conditions']:
                    mesh_info['boundary_conditions'][bc_label] = {'data': {}, 'type': bc_config.get('type')}
                mesh_info['boundary_conditions'][bc_label]['data'][field_name] = bc_data[field_name]
        
    # 4. Identify boundary nodes based on the processed conditions and mesh geometry
    identify_boundary_nodes(mesh_info, boundary_faces)

    # 5. Prepare the final output dictionary
    # The 'boundary_conditions' in the output now contains the grid indices from identify_boundary_nodes
    processed_boundary_conditions = convert_numpy_to_list(mesh_info["boundary_conditions"])

    pre_processed_output = {
        "domain_settings": domain_settings,
        "fluid_properties": input_data.get("fluid_properties", {}),
        "simulation_parameters": input_data.get("simulation_parameters", {}),
        "initial_conditions": input_data.get("initial_conditions", {}),
        "boundary_conditions": processed_boundary_conditions,
        "mesh_info": mesh_info,
        "mesh": {
            "boundary_faces": boundary_faces
        }
    }

    print("DEBUG (pre_process_input_data): Pre-processing complete. Output structure prepared.")
    return pre_processed_output

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/pre_process_input.py <input_json_filepath> <output_json_filepath>")
        print("Example: python src/pre_process_input.py data/fluid_simulation_input.json temp/solver_input.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: '{input_file}'")

        with open(input_file, 'r') as f:
            raw_input_data = json.load(f)

        validate_json_schema(raw_input_data)

        pre_processed_output = pre_process_input_data(raw_input_data)

        print("\n--- DEBUG: Pre-processed JSON to be passed to main_solver.py ---")
        # Use a more readable dump for debugging
        print(json.dumps(pre_processed_output, indent=2, default=lambda x: str(x) if isinstance(x, np.ndarray) else x))
        print("--- END DEBUG ---\n")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_json(pre_processed_output, output_file)
        print(f"Pre-processing successful! Data saved to '{output_file}'")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error during pre-processing: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during pre-processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


