# src/pre_process_input.py

import json
import sys
import os
import numpy as np
from itertools import chain

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
    if not data["mesh"]["boundary_faces"]:
        raise ValueError("'mesh.boundary_faces' list cannot be empty.")
    
    # Check for required fields in boundary_faces and node format
    for i, face in enumerate(data["mesh"]["boundary_faces"]):
        if "nodes" not in face:
            raise ValueError(f"Boundary face at index {i} is missing a 'nodes' key.")
        
        # --- MODIFIED LOGIC HERE ---
        # Now expecting 'nodes' to be a dictionary, and we'll extract its values (the actual coordinates)
        if not isinstance(face["nodes"], dict):
            raise ValueError(f"Boundary face at index {i} 'nodes' value must be a dictionary (e.g., node_id: [x,y,z]). Found type: {type(face['nodes'])}")
        
        # Validate the coordinates within the 'nodes' dictionary values
        for node_id, node_coords in face["nodes"].items():
            if not (isinstance(node_coords, list) and len(node_coords) == 3 and all(isinstance(coord, (int, float)) for coord in node_coords)):
                raise ValueError(f"Node '{node_id}' in boundary face {i} must be a 3-element list [x, y, z] of numbers.")

    # Basic fluid properties check
    fluid_props = data.get("fluid_properties", {})
    if not isinstance(fluid_props.get("density"), (int, float)) or fluid_props["density"] <= 0:
        raise ValueError("Fluid density must be a positive number.")
    if not isinstance(fluid_props.get("viscosity"), (int, float)) or fluid_props["viscosity"] < 0:
        raise ValueError("Fluid viscosity must be a non-negative number.")

    # Basic simulation parameters check
    sim_params = data.get("simulation_parameters", {})
    if not isinstance(sim_params.get("time_step"), (int, float)) or sim_params["time_step"] <= 0:
        raise ValueError("Simulation time_step must be a positive number.")
    if not isinstance(sim_params.get("total_time"), (int, float)) or sim_params["total_time"] <= 0:
        raise ValueError("Simulation total_time must be a positive number.")
    if sim_params.get("solver") not in ["explicit", "implicit"]:
        raise ValueError("Solver must be 'explicit' or 'implicit'.")

    # Basic boundary conditions check (ensure it's a dictionary)
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
        # Iterate over the VALUES (the [x,y,z] lists) of the 'nodes' dictionary
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


def infer_uniform_grid_parameters(min_val, max_val, all_coords_for_axis, axis_name):
    """
    Infers dx and nx (or dy, ny / dz, nz) for a uniform grid along a single axis.
    This function now counts unique coordinates to determine nx/ny/nz and
    calculates an average spacing, relaxing the strict uniform spacing requirement.
    """
    # Collect all coordinates for this axis from ALL boundary nodes
    unique_coords = sorted(list(set(all_coords_for_axis)))

    if not unique_coords:
        raise ValueError(f"No unique coordinates found for {axis_name}-axis in boundary faces.")

    num_cells = len(unique_coords)
    
    if num_cells < 1: # This case should theoretically not be hit if unique_coords is not empty
        raise ValueError(f"Cannot determine grid resolution for {axis_name}-axis: fewer than 1 unique coordinate.")
    
    # If num_cells is 1, it means the dimension effectively collapses (e.g., a 2D plane in 3D)
    # or the domain has zero extent along this axis.
    # In such cases, dx can be set to an arbitrary value (e.g., 1.0) and nx will be 1.
    # The solver then needs to handle this as a 2D/1D problem for that dimension.
    if num_cells == 1:
        # If min_val == max_val, the dimension is truly a point/plane.
        # Set spacing to a nominal value (e.g., 1.0) as it won't be used to define extent.
        if abs(max_val - min_val) < 1e-9: # Check for floating point equality
            spacing = 1.0
        else:
            # This case implies num_cells is 1 but max_val != min_val, which is an inconsistency
            # in how unique_coords was derived or the input data itself.
            # Fallback to the extent, but this is highly unusual for a single unique coord.
            print(f"Warning: Unexpected condition in {axis_name}-axis: num_cells=1 but max_val != min_val. Forcing spacing to (max-min).", file=sys.stderr)
            spacing = (max_val - min_val) # Spacing is the extent itself if only one unique point defined the range.
    else:
        # Calculate the average spacing based on the extent and number of unique planes (num_cells)
        # This gives us the uniform spacing for the new structured grid.
        spacing = (max_val - min_val) / (num_cells - 1)
        # Add a check for extremely small spacing which might indicate numerical issues
        if spacing < 1e-9 and abs(max_val - min_val) > 1e-9: # if spacing is near zero but domain has extent
             print(f"Warning: Extremely small calculated spacing ({spacing:.2e}) for {axis_name}-axis with significant extent. This might indicate many unique, closely clustered points.", file=sys.stderr)


    print(f"Inferred {axis_name}-axis: {num_cells} cells, average spacing {spacing:.6e}")
    return spacing, num_cells


def pre_process_input(input_data):
    """
    Pre-processes the raw input JSON data into a structured format
    suitable for the Navier-Stokes solver.
    """
    
    # 1. Extract Domain Extents
    # The get_domain_extents function has been updated to handle the dict-of-nodes format.
    min_x, max_x, min_y, max_y, min_z, max_z = get_domain_extents(input_data["mesh"]["boundary_faces"])

    # 2. Extract ALL coordinates for each axis from boundary_faces
    # This step is crucial for relaxing the strict uniformity check
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []
    for face in input_data["mesh"]["boundary_faces"]:
        # Extract values from the 'nodes' dictionary
        for node_coords in face["nodes"].values():
            all_x_coords.append(node_coords[0])
            all_y_coords.append(node_coords[1])
            all_z_coords.append(node_coords[2])

    # 3. Infer Uniform Grid Parameters for each axis based on unique planes
    dx, nx = infer_uniform_grid_parameters(min_x, max_x, all_x_coords, 'x')
    dy, ny = infer_uniform_grid_parameters(min_y, max_y, all_y_coords, 'y')
    dz, nz = infer_uniform_grid_parameters(min_z, max_z, all_z_coords, 'z')

    # Ensure nx, ny, nz are at least 1, even for effectively 2D/1D problems
    nx = max(1, nx)
    ny = max(1, ny)
    nz = max(1, nz)

    # Re-adjust dx/dy/dz if a dimension effectively collapses to a single point/plane.
    # This handles cases where min_val == max_val for a dimension, setting dx/dy/dz to 1.0,
    # and ensuring nx/ny/nz is 1. This makes the solver aware it's a 2D/1D problem
    # for that specific axis. Use a small epsilon for floating point comparison.
    TOLERANCE = 1e-9
    if abs(max_x - min_x) < TOLERANCE:
        dx = 1.0 # Set a nominal spacing as this dimension is effectively collapsed
        nx = 1
    if abs(max_y - min_y) < TOLERANCE:
        dy = 1.0
        ny = 1
    if abs(max_z - min_z) < TOLERANCE:
        dz = 1.0
        nz = 1


    domain_settings = {
        "min_x": min_x, "max_x": max_x, "dx": dx, "nx": nx,
        "min_y": min_y, "max_y": max_y, "dy": dy, "ny": ny,
        "min_z": min_z, "max_z": max_z, "dz": dz, "nz": nz
    }

    # 4. Process Boundary Conditions
    # The boundary_conditions in the input should specify rules for min_x, max_x, etc.
    # The pre-processor simply passes these on, as they are already structured.
    # No complex mapping is needed if they are defined by axis-aligned planes.
    processed_boundary_conditions = input_data.get("boundary_conditions", {})

    # 5. Extract Fluid Properties and Simulation Parameters
    fluid_properties = input_data.get("fluid_properties", {})
    simulation_parameters = input_data.get("simulation_parameters", {})

    pre_processed_data = {
        "domain_settings": domain_settings,
        "fluid_properties": fluid_properties,
        "simulation_parameters": simulation_parameters,
        "boundary_conditions": processed_boundary_conditions
    }

    return pre_processed_data


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

        # Load input data
        with open(input_file, 'r') as f:
            raw_input_data = json.load(f)

        # Validate input against a conceptual schema (simplified)
        # In a real system, you'd use a dedicated schema validation library
        validate_json_with_schema(raw_input_data, {}) # Pass an empty dict as schema since it's simplified

        # Pre-process the data
        pre_processed_output = pre_process_input(raw_input_data)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save pre-processed data to output file
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
