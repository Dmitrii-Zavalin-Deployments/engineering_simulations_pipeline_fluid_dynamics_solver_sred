# src/pre_process_input.py

import json
import numpy as np
import sys
import os # Import os for path handling

def pre_process_input(input_filepath: str, output_filepath: str):
    """
    Reads a fluid simulation input file conforming to fluid_simulation_input.schema.json,
    infers structured grid parameters (dx, dy, dz) from boundary nodes,
    maps boundary conditions to a format suitable for the structured grid solver,
    and writes the transformed data to a new JSON file.

    Args:
        input_filepath: Path to the input JSON file (fluid_simulation_input.schema.json compliant).
        output_filepath: Path to the temporary output JSON file for main_solver.py.
    
    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If the input file is not valid JSON.
        ValueError: If grid resolution cannot be inferred uniformly or if boundary faces
                    do not align with axis-aligned planes, or if schema inconsistencies exist.
    """
    
    # --- 1. Load Input Data ---
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Input file not found: '{input_filepath}'")

    try:
        with open(input_filepath, 'r') as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in '{input_filepath}': {e}", e.doc, e.pos)

    # --- 2. Extract all boundary node coordinates and store boundary face definitions ---
    all_x_coords = set()
    all_y_coords = set()
    all_z_coords = set()

    # Store boundary face definitions for later mapping
    # {face_id: {'type': ..., 'nodes': {node_id: [x,y,z]}}}
    boundary_face_geometry = {} 

    mesh_data = input_data.get('mesh')
    if not mesh_data:
        raise ValueError("Input JSON is missing the 'mesh' section.")
    
    boundary_faces_list = mesh_data.get('boundary_faces')
    if not boundary_faces_list:
        raise ValueError("Input JSON is missing 'mesh.boundary_faces' section.")

    for b_face in boundary_faces_list:
        face_id = str(b_face.get('face_id')) # Get face_id, ensure it's string for dict key
        if face_id is None:
            raise ValueError("A boundary face is missing 'face_id'.")
        if face_id in boundary_face_geometry:
            print(f"Warning: Duplicate face_id '{face_id}' found in mesh.boundary_faces. Overwriting.", file=sys.stderr)

        boundary_face_geometry[face_id] = b_face
        
        nodes_data = b_face.get('nodes')
        if not nodes_data:
            raise ValueError(f"Boundary face '{face_id}' is missing 'nodes' geometry.")

        for node_coords in nodes_data.values():
            if not (isinstance(node_coords, list) and len(node_coords) == 3 and 
                    all(isinstance(c, (int, float)) for c in node_coords)):
                raise ValueError(f"Node coordinates for face '{face_id}' are not a 3-element numeric array: {node_coords}")
            all_x_coords.add(node_coords[0])
            all_y_coords.add(node_coords[1])
            all_z_coords.add(node_coords[2])

    # --- 3. Determine Overall Domain Extents (min/max for each axis) ---
    # Handle cases where a dimension might be effectively 2D (only one unique coordinate)
    # If a set of coordinates is empty, it implies a 0-D domain, which might be an error
    if not all_x_coords or not all_y_coords or not all_z_coords:
        raise ValueError("No valid node coordinates found in boundary faces to determine domain extents. "
                         "Ensure 'mesh.boundary_faces' contains 'nodes' with [x,y,z] coordinates.")

    min_x, max_x = min(all_x_coords), max(all_x_coords)
    min_y, max_y = min(all_y_coords), max(all_y_coords)
    min_z, max_z = min(all_z_coords), max(all_z_coords)

    # --- 4. Infer dx, dy, dz from unique coordinate differences ---
    # This is the critical part that relies on the "no assumptions" constraint.
    # It assumes the boundary nodes implicitly define a perfectly uniform structured grid.
    TOLERANCE = 1e-9 # Floating point comparison tolerance

    def get_uniform_spacing_and_count(coords_set: set, axis_name: str):
        """
        Infers uniform spacing (d_axis) and number of cells (n_axis) along an axis.
        Raises ValueError if spacing is not uniform.
        Returns 0.0 and 1 for effectively 2D or 1D dimensions (where extent is zero).
        """
        sorted_coords = sorted(list(coords_set))
        num_points = len(sorted_coords)

        # If only one point or extent is effectively zero, it's a 2D/1D plane in this dimension
        if num_points <= 1 or abs(sorted_coords[-1] - sorted_coords[0]) < TOLERANCE:
            return 0.0, 1 # Spacing is 0, and there's effectively 1 'cell' in this dimension
        
        diffs = [sorted_coords[i+1] - sorted_coords[i] for i in range(num_points - 1)]
        
        # Filter out near-zero differences, which might occur if points are duplicated or too close
        significant_diffs = [d for d in diffs if d > TOLERANCE]

        if not significant_diffs:
            # This shouldn't happen if num_points > 1 and extent > TOLERANCE
            # but as a safeguard, indicates all points are effectively at the same coord
            return 0.0, 1

        # Check for uniformity: all significant differences should be "the same" within tolerance.
        first_diff = significant_diffs[0]
        for d in significant_diffs:
            if abs(d - first_diff) > TOLERANCE:
                raise ValueError(
                    f"Inconsistent spacing detected along {axis_name}-axis ({d} vs {first_diff}). "
                    "Cannot infer uniform grid resolution without assumptions. "
                    "Input mesh boundary nodes do not form a perfectly structured grid. "
                    "Please ensure nodes are uniformly spaced on axis-aligned boundaries."
                )
        
        # Calculate number of cells based on total extent and derived dx
        extent = sorted_coords[-1] - sorted_coords[0]
        # Use a small tolerance for division to avoid issues with floating point arithmetic
        num_cells = int(round(extent / first_diff)) if first_diff > TOLERANCE else 1
        return first_diff, num_cells

    dx, nx = get_uniform_spacing_and_count(all_x_coords, 'x')
    dy, ny = get_uniform_spacing_and_count(all_y_coords, 'y')
    dz, nz = get_uniform_spacing_and_count(all_z_coords, 'z')

    # --- 5. Map Boundary Conditions to the Structured Grid Format ---
    # This structure is what main_solver.py currently expects.
    converted_boundary_conditions = {
        "inlet": {},
        "outlet": {},
        "wall": {}
    }

    # Helper to check if a set of points forms a plane at a specific coordinate
    def is_on_plane(coords_map: dict, plane_coord_val: float, axis_index: int, tolerance: float = TOLERANCE) -> bool:
        """Checks if all nodes in coords_map lie on a plane defined by plane_coord_val on axis_index."""
        if not coords_map: return False 
        for coords in coords_map.values():
            if abs(coords[axis_index] - plane_coord_val) > tolerance:
                return False
        return True

    input_bcs = input_data.get('boundary_conditions')
    if not input_bcs:
        raise ValueError("Input JSON is missing the 'boundary_conditions' section.")

    # Process each boundary type (inlet, outlet, wall) from the input schema
    for bc_type, bc_data in input_bcs.items():
        if bc_type not in ["inlet", "outlet", "wall"]:
            print(f"Warning: Skipping unsupported top-level boundary condition type '{bc_type}'.", file=sys.stderr)
            continue

        if not isinstance(bc_data, dict):
            raise ValueError(f"Boundary condition data for '{bc_type}' is not an object.")

        # Copy global properties (velocity, pressure, no_slip) for the BC type
        if bc_type == "inlet":
            if "velocity" in bc_data:
                converted_boundary_conditions["inlet"]["velocity"] = bc_data["velocity"]
            if "pressure" in bc_data:
                converted_boundary_conditions["inlet"]["pressure"] = bc_data["pressure"]
        elif bc_type == "outlet":
            if "pressure" in bc_data:
                converted_boundary_conditions["outlet"]["pressure"] = bc_data["pressure"]
        elif bc_type == "wall":
            # 'no_slip' is required in schema, but good to handle potential missing with default
            converted_boundary_conditions["wall"]["no_slip"] = bc_data.get("no_slip", True)

        # Identify which structured grid planes correspond to this BC type
        faces_list = bc_data.get('faces', [])
        if not isinstance(faces_list, list):
            raise ValueError(f"Faces list for '{bc_type}' boundary condition is not an array.")

        for face_id_int in faces_list:
            face_id = str(face_id_int) # Convert to string to match dict keys

            if face_id not in boundary_face_geometry:
                raise ValueError(
                    f"Boundary face_id '{face_id}' for '{bc_type}' boundary condition "
                    "not found in 'mesh.boundary_faces'. This indicates a schema inconsistency."
                )
            
            face_info = boundary_face_geometry[face_id]
            face_nodes_coords_map = face_info.get('nodes')
            face_declared_type = face_info.get('type')

            if not face_nodes_coords_map:
                raise ValueError(f"Boundary face '{face_id}' has no 'nodes' data.")
            if face_declared_type is None:
                raise ValueError(f"Boundary face '{face_id}' is missing its 'type'.")

            # Validate that the face's type matches the boundary condition type it's listed under
            if face_declared_type != bc_type:
                raise ValueError(
                    f"Inconsistency: Boundary face_id '{face_id}' is declared as type '{face_declared_type}' "
                    f"in mesh.boundary_faces, but is listed under '{bc_type}' boundary conditions. "
                    "This is a schema violation and indicates a problem in the input data."
                )

            matched_plane = None
            if is_on_plane(face_nodes_coords_map, min_x, 0): matched_plane = "min_x"
            elif is_on_plane(face_nodes_coords_map, max_x, 0): matched_plane = "max_x"
            elif is_on_plane(face_nodes_coords_map, min_y, 1): matched_plane = "min_y"
            elif is_on_plane(face_nodes_coords_map, max_y, 1): matched_plane = "max_y"
            elif is_on_plane(face_nodes_coords_map, min_z, 2): matched_plane = "min_z"
            elif is_on_plane(face_nodes_coords_map, max_z, 2): matched_plane = "max_z"
            
            if matched_plane is None:
                raise ValueError(
                    f"Boundary face_id '{face_id}' (type '{face_declared_type}') does not correspond "
                    "to a simple axis-aligned boundary plane of the inferred structured grid. "
                    "This input cannot be processed by the current structured grid solver without making assumptions."
                )
            
            # Set the flag for the matched plane in the structured BC dictionary
            converted_boundary_conditions[bc_type][matched_plane] = True

    # --- 6. Construct the temporary JSON output for main_solver.py ---
    simulation_parameters = input_data.get("simulation_parameters")
    if not simulation_parameters:
        raise ValueError("Input JSON is missing the 'simulation_parameters' section.")

    fluid_properties = input_data.get("fluid_properties")
    if not fluid_properties:
        raise ValueError("Input JSON is missing the 'fluid_properties' section.")

    temp_output_data = {
        "domain_settings": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            # nx, ny, nz are derived from domain_settings by the current main_solver
            # but we include them here for clarity and potential direct use.
            "nx": nx, 
            "ny": ny,
            "nz": nz
        },
        "fluid_properties": fluid_properties,
        "simulation_parameters": simulation_parameters,
        "boundary_conditions": converted_boundary_conditions
    }

    # --- 7. Write the temporary JSON file ---
    try:
        with open(output_filepath, 'w') as f:
            json.dump(temp_output_data, f, indent=2)
    except IOError as e:
        raise IOError(f"Could not write output file '{output_filepath}': {e}")


    print(f"Successfully converted '{input_filepath}' to '{output_filepath}' for structured grid solver.")
    print(f"Inferred structured grid: Nx={nx}, Ny={ny}, Nz={nz} with Dx={dx:.4e}, Dy={dy:.4e}, Dz={dz:.4e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/pre_process_input.py <input_fluid_sim_json> <output_temp_json>")
        print("Example: python src/pre_process_input.py input/my_complex_fluid_sim.json temp/solver_input.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        pre_process_input(input_file, output_file)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, IOError) as e:
        print(f"Error during pre-processing: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)