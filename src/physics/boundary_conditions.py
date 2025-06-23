import numpy as np
import sys

# Assume a small tolerance for floating point comparisons
TOLERANCE = 1e-6

def identify_boundary_nodes(boundary_conditions_definition, all_mesh_boundary_faces, mesh_info):
    """
    Identifies boundary conditions by mapping face_ids to structured grid cell indices.

    Args:
        boundary_conditions_definition (dict): Dictionary from input JSON's 'boundary_conditions' section.
                                               e.g., {"inlet": {"faces": [113, 115], ...}, ...}
        all_mesh_boundary_faces (list): List of dictionaries from input JSON's 'mesh.boundary_faces' section.
                                        Each dict has 'face_id' and 'nodes'.
        mesh_info (dict): Contains 'grid_shape', 'dx', 'dy', 'dz', 'min_x', 'max_x', etc.,
                          and 'all_cell_centers_flat' (nx*ny*nz, 3 array of (x,y,z) coords for cell centers).

    Returns:
        dict: A structured dictionary optimized for vectorized boundary application.
              Example:
              {
                  "inlet": {
                      "type": "dirichlet",
                      "cell_indices": np.array([[i1, j1, k1], [i2, j2, k2], ...]),
                      "velocity": np.array([vx, vy, vz]),
                      "pressure": p_val,
                      "boundary_dim": 0, # Axis-aligned dimension (0=X, 1=Y, 2=Z) or None
                      "boundary_side": "min" # "min" or "max" for that dim
                  },
                  # ... other boundaries
              }
    """
    print("DEBUG (identify_boundary_nodes): Starting boundary node identification.")
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']
    min_x, max_x = mesh_info['min_x'], mesh_info['max_x']
    min_y, max_y = mesh_info['min_y'], mesh_info['max_y']
    min_z, max_z = mesh_info['min_z'], mesh_info['max_z']

    mesh_face_lookup = {face['face_id']: face for face in all_mesh_boundary_faces}
    print(f"DEBUG (identify_boundary_nodes): Mesh face lookup created with {len(mesh_face_lookup)} entries.")

    processed_bcs = {}

    for bc_name, bc_properties in boundary_conditions_definition.items():
        print(f"DEBUG (identify_boundary_nodes): Processing BC '{bc_name}'.")
        bc_type_from_name = bc_name
        bc_type = bc_properties.get("type")
        if not bc_type: # If 'type' not explicitly provided in JSON
            if bc_type_from_name == "inlet": bc_type = "dirichlet"
            elif bc_type_from_name == "outlet": bc_type = "pressure_outlet"
            elif bc_type_from_name == "wall": bc_type = "dirichlet"
            else:
                bc_type = "unknown"
                print(f"Warning: Unknown boundary condition type derived for '{bc_name}'. Using 'unknown'.", file=sys.stderr)
        print(f"DEBUG (identify_boundary_nodes): BC '{bc_name}' resolved type: '{bc_type}'.")

        bc_faces_ids = bc_properties.get("faces", [])
        if not bc_faces_ids:
            print(f"Warning: Boundary condition '{bc_name}' has no 'faces' specified. Skipping.", file=sys.stderr)
            continue
        print(f"DEBUG (identify_boundary_nodes): BC '{bc_name}' has {len(bc_faces_ids)} associated faces.")

        current_bc_cell_indices = set()
        overall_min_coords_boundary = np.array([np.inf, np.inf, np.inf])
        overall_max_coords_boundary = np.array([-np.inf, -np.inf, -np.inf])

        for face_id in bc_faces_ids:
            mesh_face_data = mesh_face_lookup.get(face_id)
            if not mesh_face_data:
                print(f"Warning: Mesh face with face_id {face_id} not found for boundary '{bc_name}'. Skipping.", file=sys.stderr)
                continue
            
            face_nodes_coords = np.array(list(mesh_face_data['nodes'].values()), dtype=np.float64)
            face_min_coords = np.min(face_nodes_coords, axis=0)
            face_max_coords = np.max(face_nodes_coords, axis=0)

            overall_min_coords_boundary = np.minimum(overall_min_coords_boundary, face_min_coords)
            overall_max_coords_boundary = np.maximum(overall_max_coords_boundary, face_max_coords)

            i_min_face = int(np.floor((face_min_coords[0] - min_x) / dx - TOLERANCE)) if dx > TOLERANCE else 0
            i_max_face = int(np.ceil((face_max_coords[0] - min_x) / dx + TOLERANCE)) if dx > TOLERANCE else nx
            j_min_face = int(np.floor((face_min_coords[1] - min_y) / dy - TOLERANCE)) if dy > TOLERANCE else 0
            j_max_face = int(np.ceil((face_max_coords[1] - min_y) / dy + TOLERANCE)) if dy > TOLERANCE else ny
            k_min_face = int(np.floor((face_min_coords[2] - min_z) / dz - TOLERANCE)) if dz > TOLERANCE else 0
            k_max_face = int(np.ceil((face_max_coords[2] - min_z) / dz + TOLERANCE)) if dz > TOLERANCE else nz

            i_min_face = max(0, min(nx, i_min_face))
            i_max_face = max(0, min(nx, i_max_face))
            j_min_face = max(0, min(ny, j_min_face))
            j_max_face = max(0, min(ny, j_max_face))
            k_min_face = max(0, min(nz, k_min_face))
            k_max_face = max(0, min(nz, k_max_face))
            
            # Print calculated ranges for each face
            print(f"DEBUG (identify_boundary_nodes): Face {face_id} for '{bc_name}' maps to i[{i_min_face}:{i_max_face}], j[{j_min_face}:{j_max_face}], k[{k_min_face}:{k_max_face}]")


            for i in range(i_min_face, i_max_face):
                for j in range(j_min_face, j_max_face):
                    for k in range(k_min_face, k_max_face):
                        current_bc_cell_indices.add((i, j, k))
        
        if not current_bc_cell_indices:
            print(f"Warning: No structured grid cells found for boundary '{bc_name}' matching its face definitions. Skipping.", file=sys.stderr)
            continue
        
        final_cell_indices_array = np.array(list(current_bc_cell_indices), dtype=int)
        print(f"DEBUG (identify_boundary_nodes): BC '{bc_name}' identified {final_cell_indices_array.shape[0]} unique cells.")


        boundary_dim = None
        boundary_side = None
        interior_neighbor_offset = np.array([0, 0, 0], dtype=int)

        if abs(overall_min_coords_boundary[0] - min_x) < TOLERANCE and \
           abs(overall_max_coords_boundary[0] - min_x) < TOLERANCE:
            boundary_dim = 0; boundary_side = "min"; interior_neighbor_offset[0] = 1
        elif abs(overall_min_coords_boundary[0] - max_x) < TOLERANCE and \
             abs(overall_max_coords_boundary[0] - max_x) < TOLERANCE:
            boundary_dim = 0; boundary_side = "max"; interior_neighbor_offset[0] = -1
        elif abs(overall_min_coords_boundary[1] - min_y) < TOLERANCE and \
             abs(overall_max_coords_boundary[1] - min_y) < TOLERANCE:
            boundary_dim = 1; boundary_side = "min"; interior_neighbor_offset[1] = 1
        elif abs(overall_min_coords_boundary[1] - max_y) < TOLERANCE and \
             abs(overall_max_coords_boundary[1] - max_y) < TOLERANCE:
            boundary_dim = 1; boundary_side = "max"; interior_neighbor_offset[1] = -1
        elif abs(overall_min_coords_boundary[2] - min_z) < TOLERANCE and \
             abs(overall_max_coords_boundary[2] - min_z) < TOLERANCE:
            boundary_dim = 2; boundary_side = "min"; interior_neighbor_offset[2] = 1
        elif abs(overall_min_coords_boundary[2] - max_z) < TOLERANCE and \
             abs(overall_max_coords_boundary[2] - max_z) < TOLERANCE:
            boundary_dim = 2; boundary_side = "max"; interior_neighbor_offset[2] = -1
        
        if boundary_dim is None:
            print(f"Warning: Boundary '{bc_name}' with faces {bc_faces_ids} does not perfectly align with a single external axis-aligned plane of the structured grid. "
                  "Neumann/Pressure Outlet conditions may behave unexpectedly. Treating as generic cells for now.", file=sys.stderr)


        apply_to = bc_properties.get("apply_to", ["velocity", "pressure"])
        if bc_name == "wall" and bc_properties.get("no_slip", False):
            apply_to = ["velocity"]

        processed_bcs[bc_name] = {
            "type": bc_type,
            "cell_indices": final_cell_indices_array,
            "velocity": np.array(bc_properties.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
            "pressure": bc_properties.get("pressure", 0.0),
            "apply_to": apply_to,
            "boundary_dim": boundary_dim,
            "boundary_side": boundary_side,
            "interior_neighbor_offset": interior_neighbor_offset
        }
    
    print("DEBUG (identify_boundary_nodes): Finished boundary node identification. Returning processed_bcs.")
    return processed_bcs


def apply_boundary_conditions(velocity_field, pressure_field, fluid_properties, mesh_info, is_tentative_step):
    """
    Applies boundary conditions to the velocity and pressure fields using NumPy indexing.
    This function modifies the input fields in-place.

    Args:
        velocity_field (np.ndarray): The current velocity field (nx, ny, nz, 3).
        pressure_field (np.ndarray): The current pressure field (nx, ny, nz).
        fluid_properties (dict): Dictionary with fluid properties (e.g., "density", "viscosity").
        mesh_info (dict): Contains 'grid_shape', 'dx', 'dy', 'dz', and 'boundary_conditions' processed by identify_boundary_nodes.
        is_tentative_step (bool): True if applying BCs to u* (tentative velocity), False for final u^(n+1) or pressure.
    """
    print(f"DEBUG (apply_boundary_conditions): Function called. is_tentative_step={is_tentative_step}")
    
    # Check if inputs are valid numpy arrays
    if not isinstance(velocity_field, np.ndarray) or velocity_field.dtype != np.float64:
        print(f"ERROR (apply_boundary_conditions): velocity_field is not a float64 numpy array or is None. Type: {type(velocity_field)}, Dtype: {getattr(velocity_field, 'dtype', 'N/A')}", file=sys.stderr)
        # We must return *something* unpackable if this occurs and we don't sys.exit
        return velocity_field, pressure_field
    if not isinstance(pressure_field, np.ndarray) or pressure_field.dtype != np.float64:
        print(f"ERROR (apply_boundary_conditions): pressure_field is not a float64 numpy array or is None. Type: {type(pressure_field)}, Dtype: {getattr(pressure_field, 'dtype', 'N/A')}", file=sys.stderr)
        return velocity_field, pressure_field # Return existing, potentially invalid fields

    print(f"DEBUG (apply_boundary_conditions): velocity_field shape: {velocity_field.shape}, pressure_field shape: {pressure_field.shape}")

    processed_bcs = mesh_info.get('boundary_conditions')
    if processed_bcs is None:
        print("ERROR (apply_boundary_conditions): 'boundary_conditions' not found in mesh_info. This is critical.", file=sys.stderr)
        # If processed_bcs is None, the loop won't run. We still need to return.
        return velocity_field, pressure_field
    
    print(f"DEBUG (apply_boundary_conditions): Number of processed_bcs found: {len(processed_bcs)}")
    
    nx, ny, nz = mesh_info['grid_shape']

    for bc_name, bc in processed_bcs.items():
        print(f"DEBUG (apply_boundary_conditions): Processing BC '{bc_name}'. Type: '{bc['type']}'.")
        bc_type = bc["type"]
        cell_indices = bc["cell_indices"]
        target_velocity = bc["velocity"]
        target_pressure = bc["pressure"]
        apply_to_fields = bc["apply_to"]
        boundary_dim = bc["boundary_dim"]
        interior_neighbor_offset = bc["interior_neighbor_offset"]

        if cell_indices.size == 0:
            print(f"DEBUG (apply_boundary_conditions): Warning: No cells found for boundary '{bc_name}'. Skipping application.", file=sys.stderr)
            continue
        
        print(f"DEBUG (apply_boundary_conditions): Applying to {cell_indices.shape[0]} cells for '{bc_name}'.")

        # Dirichlet (Fixed Value) Boundary Conditions
        if bc_type == "dirichlet":
            print(f"DEBUG (apply_boundary_conditions): Applying Dirichlet BC for '{bc_name}'.")
            if "velocity" in apply_to_fields:
                print(f"DEBUG (apply_boundary_conditions): Setting velocity for '{bc_name}' to {target_velocity}.")
                velocity_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_velocity
            if "pressure" in apply_to_fields and not is_tentative_step:
                print(f"DEBUG (apply_boundary_conditions): Setting pressure for '{bc_name}' to {target_pressure} (non-tentative step).")
                pressure_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_pressure

        # Neumann (Fixed Gradient/Outflow) Boundary Conditions - Zero Gradient
        elif bc_type == "neumann":
            print(f"DEBUG (apply_boundary_conditions): Applying Neumann BC for '{bc_name}'.")
            if boundary_dim is not None:
                neighbor_indices = cell_indices + interior_neighbor_offset
                valid_neighbors_mask = (neighbor_indices[:,0] >= 0) & (neighbor_indices[:,0] < nx) & \
                                       (neighbor_indices[:,1] >= 0) & (neighbor_indices[:,1] < ny) & \
                                       (neighbor_indices[:,2] >= 0) & (neighbor_indices[:,2] < nz)

                valid_cell_indices = cell_indices[valid_neighbors_mask]
                valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
                
                print(f"DEBUG (apply_boundary_conditions): Neumann valid cells: {valid_cell_indices.shape[0]}.")

                if valid_cell_indices.size > 0:
                    if "velocity" in apply_to_fields:
                        print(f"DEBUG (apply_boundary_conditions): Copying velocity from neighbors for '{bc_name}'.")
                        velocity_field[valid_cell_indices[:,0], valid_cell_indices[:,1], valid_cell_indices[:,2]] = \
                            velocity_field[valid_neighbor_indices[:,0], valid_neighbor_indices[:,1], valid_neighbor_indices[:,2]]

                    if "pressure" in apply_to_fields and not is_tentative_step:
                        print(f"DEBUG (apply_boundary_conditions): Copying pressure from neighbors for '{bc_name}' (non-tentative step).")
                        pressure_field[valid_cell_indices[:,0], valid_cell_indices[:,1], valid_cell_indices[:,2]] = \
                            pressure_field[valid_neighbor_indices[:,0], valid_neighbor_indices[:,1], valid_neighbor_indices[:,2]]
                else:
                    print(f"Warning: No valid interior neighbors found for Neumann BC '{bc_name}'. Check grid size/boundary alignment.", file=sys.stderr)
            else:
                print(f"Warning: Neumann BC '{bc_name}' on non-axis-aligned external boundary or internal face. Skipping application. Manual handling required.", file=sys.stderr)


        # Pressure Outlet Boundary Condition (often combined with zero-gradient velocity)
        elif bc_type == "pressure_outlet":
            print(f"DEBUG (apply_boundary_conditions): Applying Pressure Outlet BC for '{bc_name}'.")
            if "pressure" in apply_to_fields and not is_tentative_step:
                print(f"DEBUG (apply_boundary_conditions): Setting pressure for '{bc_name}' to {target_pressure} (non-tentative step).")
                pressure_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_pressure

            if "velocity" in apply_to_fields:
                if boundary_dim is not None:
                    neighbor_indices = cell_indices + interior_neighbor_offset

                    valid_neighbors_mask = (neighbor_indices[:,0] >= 0) & (neighbor_indices[:,0] < nx) & \
                                           (neighbor_indices[:,1] >= 0) & (neighbor_indices[:,1] < ny) & \
                                           (neighbor_indices[:,2] >= 0) & (neighbor_indices[:,2] < nz)

                    valid_cell_indices = cell_indices[valid_neighbors_mask]
                    valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
                    
                    print(f"DEBUG (apply_boundary_conditions): Pressure Outlet velocity valid cells: {valid_cell_indices.shape[0]}.")

                    if valid_cell_indices.size > 0:
                        print(f"DEBUG (apply_boundary_conditions): Copying velocity from neighbors for Pressure Outlet '{bc_name}'.")
                        velocity_field[valid_cell_indices[:,0], valid_cell_indices[:,1], valid_cell_indices[:,2]] = \
                            velocity_field[valid_neighbor_indices[:,0], valid_neighbor_indices[:,1], valid_neighbor_indices[:,2]]
                    else:
                        print(f"Warning: No valid interior neighbors found for velocity at Pressure Outlet BC '{bc_name}'. Check grid size/boundary alignment.", file=sys.stderr)
                else:
                    print(f"Warning: Pressure Outlet BC '{bc_name}' on non-axis-aligned external boundary or internal face for velocity. Skipping application. Manual handling required.", file=sys.stderr)
        else:
            print(f"Warning: Unknown boundary condition type '{bc_type}' for '{bc_name}'. Skipping.", file=sys.stderr)

    print("DEBUG (apply_boundary_conditions): All boundary conditions processed. Returning velocity_field and pressure_field.")
    return velocity_field, pressure_field
