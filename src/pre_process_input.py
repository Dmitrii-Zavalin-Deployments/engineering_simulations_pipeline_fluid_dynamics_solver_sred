# src/pre_process_input.py

import json
import sys
import os
import numpy as np
from itertools import chain

# --- FIX FOR ImportError: attempted relative import with no known parent package ---
# Get the directory where the current script (pre_process_input.py) resides.
# This will typically be /home/runner/work/.../src/
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add this directory (which contains 'physics' sub-directory) to sys.path
# so that absolute imports like 'physics.boundary_conditions' work.
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- END FIX ---

# Now, import using the absolute path from the added sys.path entry
# (i.e., relative to the 'src' directory)
from physics.boundary_conditions import identify_boundary_nodes


def load_json_schema(filepath):
    """Loads a JSON schema from a file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON schema file at {filepath}", file=sys.stderr)
        sys.exit(1)

def validate_json_with_schema(data, schema):
    """
    Validates JSON data against a given schema.
    (Simplified validation - a real implementation would use a library like jsonschema)
    """
    # Check for top-level keys
    required_keys = ["fluid_properties", "simulation_parameters", "mesh", "boundary_conditions"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Input JSON missing top-level key: '{key}'")

    # Basic mesh structure check
    if "mesh" not in data or "boundary_faces" not in data["mesh"]:
        raise ValueError("Input JSON missing 'mesh.boundary_faces' section.")
    if not isinstance(data["mesh"]["boundary_faces"], list):
        raise ValueError("'mesh.boundary_faces' must be a list.")

    for i, face in enumerate(data["mesh"]["boundary_faces"]):
        if "nodes" not in face:
            raise ValueError(f"Boundary face at index {i} is missing a 'nodes' key.")

        if not isinstance(face["nodes"], dict):
            raise ValueError(f"Boundary face at index {i} 'nodes' value must be a dictionary (e.g., node_id: [x,y,z]). Found type: {type(face['nodes'])}")

        for node_id, node_coords in face["nodes"].items():
            if not (isinstance(node_coords, list) and len(node_coords) == 3 and all(isinstance(coord, (int, float)) for coord in node_coords)):
                raise ValueError(f"Node '{node_id}' in boundary face {i} must be a 3-element list [x, y, z] of numbers.")

    fluid_props = data.get("fluid_properties", {})
    if not isinstance(fluid_props.get("density"), (int, float)) or fluid_props["density"] <= 0:
        raise ValueError("Fluid density must be a positive number.")
    if not isinstance(fluid_props.get("viscosity"), (int, float)) or fluid_props["viscosity"] < 0:
        raise ValueError("Fluid viscosity must be a non-negative number.")

    sim_params = data.get("simulation_parameters", {})
    if not isinstance(sim_params.get("time_step"), (int, float)) or sim_params["time_step"] <= 0:
        raise ValueError("Simulation time_step must be a positive number.")
    if not isinstance(sim_params.get("total_time"), (int, float)) or sim_params["total_time"] <= 0:
        raise ValueError("Simulation total_time must be a positive number.")
    if sim_params.get("solver") not in ["explicit", "implicit"]:
        raise ValueError("Solver must be 'explicit' or 'implicit'.")

    if not isinstance(data.get("boundary_conditions"), dict):
        raise ValueError("Boundary conditions must be a dictionary.")

    print("✅ Input JSON passed basic structural validation.")


def get_domain_extents(boundary_faces):
    """
    Extracts the min/max x, y, z coordinates from all nodes in boundary_faces.
    Nodes are expected to be in a dictionary format: {node_id: [x,y,z]}.
    """
    all_x = []
    all_y = []
    all_z = []

    for face in boundary_faces:
        for node_coords in face["nodes"].values():
            all_x.append(node_coords[0])
            all_y.append(node_coords[1])
            all_z.append(node_coords[2])

    if not all_x or not all_y or not all_z:
        raise ValueError("No nodes found in boundary faces to determine domain extents.")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    min_z, max_z = min(all_z), max(all_z)

    return min_x, max_x, min_y, max_y, min_z, max_z


def infer_uniform_grid_parameters(min_val, max_val, all_coords_for_axis, axis_name, json_nx=None):
    """
    Infers dx and nx (or dy, ny / dz, nz) for a uniform grid along a single axis.
    Prioritizes JSON nx/ny/nz if provided and consistent with domain extent.
    """
    # Use the nx/ny/nz from JSON's domain_definition if available
    if json_nx is not None and json_nx > 0:
        inferred_dx = (max_val - min_val) / json_nx if abs(max_val - min_val) > 1e-9 else 1.0
        inferred_num_cells = json_nx
        print(f"DEBUG (infer_uniform_grid_parameters): Using JSON specified {axis_name}x={json_nx} cells. dx={inferred_dx:.6e}")
        return inferred_dx, inferred_num_cells

    # Fallback to inferring from unique boundary face coordinates if JSON nx is not provided or invalid
    unique_coords = sorted(list(set(all_coords_for_axis)))

    if not unique_coords:
        raise ValueError(f"No unique coordinates found for {axis_name}-axis in boundary faces.")

    num_unique_planes = len(unique_coords)

    if num_unique_planes == 1:
        if abs(max_val - min_val) < 1e-9:
            spacing = 1.0
            num_cells = 1
        else:
            print(f"Warning: Unexpected condition in {axis_name}-axis: 1 unique coord but max_val != min_val. Forcing {axis_name}x=1 cell and spacing=(max-min).", file=sys.stderr)
            spacing = (max_val - min_val)
            num_cells = 1
    else:
        spacing = (max_val - min_val) / (num_unique_planes - 1)
        num_cells = num_unique_planes - 1
        
        if num_cells <= 0:
            if abs(max_val - min_val) > 1e-9:
                num_cells = 1
                spacing = (max_val - min_val)
                print(f"Warning: Inferred {axis_name}x resulted in 0 or less cells despite domain extent. Forcing {axis_name}x=1 and spacing=(max-min).", file=sys.stderr)
            else:
                num_cells = 1
                spacing = 1.0
                print(f"Warning: Inferred {axis_name}x resulted in 0 or less cells for zero-extent domain. Forcing {axis_name}x=1 and nominal spacing.", file=sys.stderr)

        if spacing < 1e-9 and abs(max_val - min_val) > 1e-9:
            print(f"Warning: Extremely small calculated spacing ({spacing:.2e}) for {axis_name}-axis with significant extent. This might indicate many unique, closely clustered points.", file=sys.stderr)

    print(f"Inferred {axis_name}-axis: {num_cells} cells, spacing {spacing:.6e}")
    return spacing, num_cells

def convert_numpy_to_list(obj):
    """
    Recursively converts NumPy arrays within a dictionary or list to standard Python lists.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # This is the key conversion
    else:
        return obj

def pre_process_input_data(input_data):
    """
    Pre-processes the raw input JSON data into a structured format
    suitable for the Navier-Stokes solver.
    """
    print("DEBUG (pre_process_input_data): Starting pre-processing.")

    min_x, max_x, min_y, max_y, min_z, max_z = get_domain_extents(input_data["mesh"]["boundary_faces"])

    all_x_coords = []
    all_y_coords = []
    all_z_coords = []
    for face in input_data["mesh"]["boundary_faces"]:
        for node_coords in face["nodes"].values():
            all_x_coords.append(node_coords[0])
            all_y_coords.append(node_coords[1])
            all_z_coords.append(node_coords[2])

    json_domain_def = input_data.get('domain_definition', {})
    json_nx = json_domain_def.get('nx')
    json_ny = json_domain_def.get('ny')
    json_nz = json_domain_def.get('nz')

    dx, nx = infer_uniform_grid_parameters(min_x, max_x, all_x_coords, 'x', json_nx)
    dy, ny = infer_uniform_grid_parameters(min_y, max_y, all_y_coords, 'y', json_ny)
    dz, nz = infer_uniform_grid_parameters(min_z, max_z, all_z_coords, 'z', json_nz)

    nx = max(1, nx)
    ny = max(1, ny)
    nz = max(1, nz)

    TOLERANCE_ZERO_EXTENT = 1e-9
    if abs(max_x - min_x) < TOLERANCE_ZERO_EXTENT:
        dx = 1.0
        nx = 1
    if abs(max_y - min_y) < TOLERANCE_ZERO_EXTENT:
        dy = 1.0
        ny = 1
    if abs(max_z - min_z) < TOLERANCE_ZERO_EXTENT:
        dz = 1.0
        nz = 1


    domain_settings = {
        "min_x": min_x, "max_x": max_x, "dx": dx, "nx": nx,
        "min_y": min_y, "max_y": max_y, "dy": dy, "ny": ny,
        "min_z": min_z, "max_z": max_z, "dz": dz, "nz": nz
    }

    all_mesh_boundary_faces = input_data["mesh"]["boundary_faces"]
    
    mesh_info_for_bc = {
        'grid_shape': (nx, ny, nz),
        'dx': dx, 'dy': dy, 'dz': dz,
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'min_z': min_z, 'max_z': max_z
    }

    processed_boundary_conditions = identify_boundary_nodes(
        input_data.get("boundary_conditions", {}),
        all_mesh_boundary_faces,
        mesh_info_for_bc
    )

    # --- START OF FIX ---
    # Recursively convert any NumPy arrays in the processed boundary conditions
    # to standard Python lists, as NumPy arrays are not JSON serializable.
    processed_boundary_conditions = convert_numpy_to_list(processed_boundary_conditions)
    # --- END OF FIX ---

    fluid_properties = input_data.get("fluid_properties", {})
    simulation_parameters = input_data.get("simulation_parameters", {})

    pre_processed_output = {
        "domain_settings": domain_settings,
        "fluid_properties": fluid_properties,
        "simulation_parameters": simulation_parameters,
        "boundary_conditions": processed_boundary_conditions,
        "mesh": {
            "boundary_faces": all_mesh_boundary_faces
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

        validate_json_with_schema(raw_input_data, {})

        pre_processed_output = pre_process_input_data(raw_input_data)

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w') as f:
            json.dump(pre_processed_output, f, indent=2)
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



