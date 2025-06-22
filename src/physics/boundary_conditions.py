import numpy as np
import sys # Added for printing warnings to stderr

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
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']
    # Grid lines are needed for converting world coords to cell indices
    min_x, max_x = mesh_info['min_x'], mesh_info['max_x']
    min_y, max_y = mesh_info['min_y'], mesh_info['max_y']
    min_z, max_z = mesh_info['min_z'], mesh_info['max_z']

    # Create a lookup for mesh faces by their face_id for quick access
    mesh_face_lookup = {face['face_id']: face for face in all_mesh_boundary_faces}

    processed_bcs = {}

    # Iterate through high-level boundary definitions (e.g., "inlet", "outlet", "wall")
    for bc_name, bc_properties in boundary_conditions_definition.items():
        # Determine actual physical type based on the name if not explicitly provided, or from properties
        # Assuming common mappings: "inlet" -> dirichlet, "outlet" -> pressure_outlet, "wall" -> dirichlet
        bc_type_from_name = bc_name # Use the key name as the initial type guess
        if "type" in bc_properties: # Allow explicit type override in JSON
            bc_type = bc_properties["type"]
        elif bc_type_from_name == "inlet": bc_type = "dirichlet"
        elif bc_type_from_name == "outlet": bc_type = "pressure_outlet" # Common for outlets
        elif bc_type_from_name == "wall": bc_type = "dirichlet" # For no-slip walls
        else:
            bc_type = "unknown" # Fallback for unhandled types
            print(f"Warning: Unknown boundary condition type derived for '{bc_name}'. Using 'unknown'.", file=sys.stderr)


        bc_faces_ids = bc_properties.get("faces", [])
        if not bc_faces_ids:
            print(f"Warning: Boundary condition '{bc_name}' has no 'faces' specified. Skipping.", file=sys.stderr)
            continue

        # Accumulate all cell (i,j,k) indices for this logical boundary (e.g., all cells of the "inlet")
        current_bc_cell_indices = set() # Use a set to avoid duplicate indices
        
        # Track the overall bounding box of all faces belonging to this boundary condition
        # This is used to determine if the combined boundary is axis-aligned and which side it's on.
        overall_min_coords_boundary = np.array([np.inf, np.inf, np.inf])
        overall_max_coords_boundary = np.array([-np.inf, -np.inf, -np.inf])

        for face_id in bc_faces_ids:
            mesh_face_data = mesh_face_lookup.get(face_id)
            if not mesh_face_data:
                print(f"Warning: Mesh face with face_id {face_id} not found for boundary '{bc_name}'. Skipping.", file=sys.stderr)
                continue
            
            # Extract node coordinates for the current mesh face
            # Nodes are stored as a dictionary {node_id: [x,y,z]}, get the values
            face_nodes_coords = np.array(list(mesh_face_data['nodes'].values()), dtype=np.float64)
            
            # Calculate the bounding box for THIS INDIVIDUAL MESH FACE
            face_min_coords = np.min(face_nodes_coords, axis=0)
            face_max_coords = np.max(face_nodes_coords, axis=0)

            # Update the overall bounding box for this logical boundary (across all its face_ids)
            overall_min_coords_boundary = np.minimum(overall_min_coords_boundary, face_min_coords)
            overall_max_coords_boundary = np.maximum(overall_max_coords_boundary, face_max_coords)

            # Map coordinates to cell indices (i, j, k) for this specific mesh face
            # We determine the min/max grid indices covered by this face's bounding box.
            # Using cell center logic from pre_process_input, dx, dy, dz are cell sizes.
            # Grid index for a point 'coord' is typically (coord - min_domain) / cell_size
            
            # Adjust ranges for integer indexing. max_coords for cells should be (N-1)
            # Add a small buffer for floating point comparisons when determining bounds.
            i_min_face = int(np.floor((face_min_coords[0] - min_x) / dx - TOLERANCE)) if dx > TOLERANCE else 0
            i_max_face = int(np.ceil((face_max_coords[0] - min_x) / dx + TOLERANCE)) if dx > TOLERANCE else nx
            j_min_face = int(np.floor((face_min_coords[1] - min_y) / dy - TOLERANCE)) if dy > TOLERANCE else 0
            j_max_face = int(np.ceil((face_max_coords[1] - min_y) / dy + TOLERANCE)) if dy > TOLERANCE else ny
            k_min_face = int(np.floor((face_min_coords[2] - min_z) / dz - TOLERANCE)) if dz > TOLERANCE else 0
            k_max_face = int(np.ceil((face_max_coords[2] - min_z) / dz + TOLERANCE)) if dz > TOLERANCE else nz

            # Clamp indices to valid grid range [0, N-1]
            i_min_face = max(0, min(nx, i_min_face))
            i_max_face = max(0, min(nx, i_max_face))
            j_min_face = max(0, min(ny, j_min_face))
            j_max_face = max(0, min(ny, j_max_face))
            k_min_face = max(0, min(nz, k_min_face))
            k_max_face = max(0, min(nz, k_max_face))

            # Iterate over the determined cell index range and add to the set
            for i in range(i_min_face, i_max_face): # Note: range is exclusive upper bound
                for j in range(j_min_face, j_max_face):
                    for k in range(k_min_face, k_max_face):
                        current_bc_cell_indices.add((i, j, k))

        if not current_bc_cell_indices:
            print(f"Warning: No structured grid cells found for boundary '{bc_name}' matching its face definitions. Skipping.", file=sys.stderr)
            continue
        
        # Convert set of tuples to a NumPy array for easy indexing
        final_cell_indices_array = np.array(list(current_bc_cell_indices), dtype=int)

        # --- Determine the overall boundary dimension and side for Neumann/Outflow ---
        # This logic assumes the combined faces for a boundary condition form a single axis-aligned plane
        boundary_dim = None # 0 for X, 1 for Y, 2 for Z
        boundary_side = None # "min" or "max"
        interior_neighbor_offset = np.array([0, 0, 0], dtype=int) # Offset to find interior neighbor from boundary cell

        # Check for alignment with min/max domain boundaries
        if abs(overall_min_coords_boundary[0] - min_x) < TOLERANCE and \
           abs(overall_max_coords_boundary[0] - min_x) < TOLERANCE: # Min X plane (left boundary)
            boundary_dim = 0
            boundary_side = "min"
            interior_neighbor_offset[0] = 1 # Neighbor is (i+1, j, k)
        elif abs(overall_min_coords_boundary[0] - max_x) < TOLERANCE and \
             abs(overall_max_coords_boundary[0] - max_x) < TOLERANCE: # Max X plane (right boundary)
            boundary_dim = 0
            boundary_side = "max"
            interior_neighbor_offset[0] = -1 # Neighbor is (i-1, j, k)
        
        elif abs(overall_min_coords_boundary[1] - min_y) < TOLERANCE and \
             abs(overall_max_coords_boundary[1] - min_y) < TOLERANCE: # Min Y plane (bottom boundary)
            boundary_dim = 1
            boundary_side = "min"
            interior_neighbor_offset[1] = 1
        elif abs(overall_min_coords_boundary[1] - max_y) < TOLERANCE and \
             abs(overall_max_coords_boundary[1] - max_y) < TOLERANCE: # Max Y plane (top boundary)
            boundary_dim = 1
            boundary_side = "max"
            interior_neighbor_offset[1] = -1
        
        elif abs(overall_min_coords_boundary[2] - min_z) < TOLERANCE and \
             abs(overall_max_coords_boundary[2] - min_z) < TOLERANCE: # Min Z plane (front boundary)
            boundary_dim = 2
            boundary_side = "min"
            interior_neighbor_offset[2] = 1
        elif abs(overall_min_coords_boundary[2] - max_z) < TOLERANCE and \
             abs(overall_max_coords_boundary[2] - max_z) < TOLERANCE: # Max Z plane (back boundary)
            boundary_dim = 2
            boundary_side = "max"
            interior_neighbor_offset[2] = -1

        if boundary_dim is None:
            print(f"Warning: Boundary '{bc_name}' with faces {bc_faces_ids} does not perfectly align with a single external axis-aligned plane of the structured grid. "
                  "Neumann/Pressure Outlet conditions may behave unexpectedly. Treating as generic cells for now.", file=sys.stderr)
            # For these cases, interior_neighbor_offset remains [0,0,0], effectively meaning no specific interior neighbor for Neumann.

        # Determine `apply_to` field based on `no_slip` for walls
        apply_to = bc_properties.get("apply_to", ["velocity", "pressure"])
        if bc_name == "wall" and bc_properties.get("no_slip", False):
            # For no-slip walls, velocity is usually zero (Dirichlet), pressure is Neumann
            apply_to = ["velocity"] # Pressure is typically Neumann for walls in N-S, handled by the solver.
                                    # If pressure needs explicit BC, it's typically dP/dn=0 (Neumann)
                                    # or adjusted implicitly. For simplicity, we apply velocity here.
                                    # The implicit solver would then resolve pressure.


        processed_bcs[bc_name] = {
            "type": bc_type,
            "cell_indices": final_cell_indices_array, # Store the array of (i,j,k) tuples
            "velocity": np.array(bc_properties.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
            "pressure": bc_properties.get("pressure", 0.0), # For Dirichlet pressure
            "apply_to": apply_to, # What fields to apply to
            "boundary_dim": boundary_dim, # Which dimension the boundary is on (0, 1, 2) or None
            "boundary_side": boundary_side, # "min" or "max" side for that dimension
            "interior_neighbor_offset": interior_neighbor_offset # For calculating neighbor indices for Neumann
        }

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
    processed_bcs = mesh_info['boundary_conditions']
    nx, ny, nz = mesh_info['grid_shape']

    for bc_name, bc in processed_bcs.items():
        bc_type = bc["type"]
        cell_indices = bc["cell_indices"] # (N, 3) array of (i,j,k) tuples
        target_velocity = bc["velocity"]
        target_pressure = bc["pressure"]
        apply_to_fields = bc["apply_to"]
        boundary_dim = bc["boundary_dim"]
        # boundary_side = bc["boundary_side"] # Not used directly in apply logic
        interior_neighbor_offset = bc["interior_neighbor_offset"]

        # Ensure cell_indices is not empty
        if cell_indices.size == 0:
            print(f"Warning: No cells found for boundary '{bc_name}'. Skipping application.", file=sys.stderr)
            continue

        # Dirichlet (Fixed Value) Boundary Conditions
        if bc_type == "dirichlet":
            if "velocity" in apply_to_fields:
                velocity_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_velocity
            if "pressure" in apply_to_fields and not is_tentative_step:
                # Apply pressure Dirichlet only for final pressure, not tentative velocity step
                pressure_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_pressure

        # Neumann (Fixed Gradient/Outflow) Boundary Conditions - Zero Gradient
        elif bc_type == "neumann":
            # For zero-gradient Neumann, set boundary cell value equal to adjacent interior cell value.
            # This applies to velocity and potentially pressure (outflow).
            if boundary_dim is not None: # Only for axis-aligned external boundaries
                # Calculate interior neighbor indices
                neighbor_indices = cell_indices + interior_neighbor_offset
                
                # Filter out-of-bounds neighbor indices (e.g., if nx, ny, nz = 1 for a dimension)
                valid_neighbors_mask = (neighbor_indices[:,0] >= 0) & (neighbor_indices[:,0] < nx) & \
                                       (neighbor_indices[:,1] >= 0) & (neighbor_indices[:,1] < ny) & \
                                       (neighbor_indices[:,2] >= 0) & (neighbor_indices[:,2] < nz)
                
                valid_cell_indices = cell_indices[valid_neighbors_mask]
                valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]

                if valid_cell_indices.size > 0:
                    if "velocity" in apply_to_fields:
                        velocity_field[valid_cell_indices[:,0], valid_cell_indices[:,1], valid_cell_indices[:,2]] = \
                            velocity_field[valid_neighbor_indices[:,0], valid_neighbor_indices[:,1], valid_neighbor_indices[:,2]]

                    if "pressure" in apply_to_fields and not is_tentative_step:
                        pressure_field[valid_cell_indices[:,0], valid_cell_indices[:,1], valid_cell_indices[:,2]] = \
                            pressure_field[valid_neighbor_indices[:,0], valid_neighbor_indices[:,1], valid_neighbor_indices[:,2]]
                else:
                    print(f"Warning: No valid interior neighbors found for Neumann BC '{bc_name}'. Check grid size/boundary alignment.", file=sys.stderr)

            else:
                print(f"Warning: Neumann BC '{bc_name}' on non-axis-aligned external boundary or internal face. Skipping application. Manual handling required.", file=sys.stderr)


        # Pressure Outlet Boundary Condition (often combined with zero-gradient velocity)
        elif bc_type == "pressure_outlet":
            if "pressure" in apply_to_fields and not is_tentative_step:
                pressure_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_pressure

            # For velocity, usually zero-gradient outflow is applied, similar to Neumann
            if "velocity" in apply_to_fields:
                if boundary_dim is not None: # Only for axis-aligned external boundaries
                    neighbor_indices = cell_indices + interior_neighbor_offset
                    
                    valid_neighbors_mask = (neighbor_indices[:,0] >= 0) & (neighbor_indices[:,0] < nx) & \
                                           (neighbor_indices[:,1] >= 0) & (neighbor_indices[:,1] < ny) & \
                                           (neighbor_indices[:,2] >= 0) & (neighbor_indices[:,2] < nz)
                    
                    valid_cell_indices = cell_indices[valid_neighbors_mask]
                    valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]

                    if valid_cell_indices.size > 0:
                        velocity_field[valid_cell_indices[:,0], valid_cell_indices[:,1], valid_cell_indices[:,2]] = \
                            velocity_field[valid_neighbor_indices[:,0], valid_neighbor_indices[:,1], valid_neighbor_indices[:,2]]
                    else:
                        print(f"Warning: No valid interior neighbors found for velocity at Pressure Outlet BC '{bc_name}'. Check grid size/boundary alignment.", file=sys.stderr)
                else:
                    print(f"Warning: Pressure Outlet BC '{bc_name}' on non-axis-aligned external boundary or internal face for velocity. Skipping application. Manual handling required.", file=sys.stderr)
        else:
            print(f"Warning: Unknown boundary condition type '{bc_type}' for '{bc_name}'. Skipping.", file=sys.stderr)
