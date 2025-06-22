import numpy as np

def identify_boundary_nodes(boundary_conditions_data, mesh_info):
    """
    Identifies boundary conditions based on mesh_info and input data.
    This function will now focus on preparing direct indices/slices
    for vectorized application in apply_boundary_conditions.

    Args:
        boundary_conditions_data (dict): Raw boundary conditions from input JSON.
        mesh_info (dict): Contains 'grid_shape', 'x_coords_grid_lines', etc.

    Returns:
        dict: A structured dictionary optimized for vectorized boundary application.
              Example structure:
              {
                  "inlet": {
                      "type": "dirichlet",
                      "x_slice": (0, 1), # Corresponds to x_idx = 0
                      "y_slice": (None, None), # Full range
                      "z_slice": (None, None), # Full range
                      "velocity": np.array([vx, vy, vz]),
                      "pressure": p_val
                  },
                  "outlet": {
                      "type": "neumann",
                      "x_slice": (nx-1, nx), # Corresponds to x_idx = nx-1
                      # ... etc.
                  },
                  # ... other boundaries
              }
    """
    nx, ny, nz = mesh_info['grid_shape']
    x_coords = mesh_info['x_coords_grid_lines']
    y_coords = mesh_info['y_coords_grid_lines']
    z_coords = mesh_info['z_coords_grid_lines']

    processed_bcs = {}

    for bc_name, bc_data in boundary_conditions_data.items():
        bc_type = bc_data.get("type", "unknown")
        face_data = bc_data["face"]

        # Determine the slice for each dimension based on face definition
        x_min, x_max = face_data.get("min_x", x_coords[0]), face_data.get("max_x", x_coords[-1])
        y_min, y_max = face_data.get("min_y", y_coords[0]), face_data.get("max_y", y_coords[-1])
        z_min, z_max = face_data.get("min_z", z_coords[0]), face_data.get("max_z", z_coords[-1])
        y_pos = face_data.get("y_pos") # Specific for 2D plane in Y

        x_slice = None
        y_slice = None
        z_slice = None
        adjacent_slice = None # To store slice for adjacent interior cells for Neumann BCs

        # Identify boundary index based on coordinate. Use a small tolerance for float comparison.
        tolerance = 1e-9

        if abs(x_min - x_coords[0]) < tolerance and abs(x_max - x_coords[0]) < tolerance:
            x_slice = (0, 1) # First X plane (left boundary)
            adjacent_slice = (1, 2) # Adjacent interior plane
        elif abs(x_min - x_coords[-1]) < tolerance and abs(x_max - x_coords[-1]) < tolerance:
            x_slice = (nx - 1, nx) # Last X plane (right boundary)
            adjacent_slice = (nx - 2, nx - 1) # Adjacent interior plane
        elif abs(y_min - y_coords[0]) < tolerance and abs(y_max - y_coords[0]) < tolerance:
            y_slice = (0, 1) # First Y plane (bottom boundary)
            adjacent_slice = (1, 2)
        elif abs(y_min - y_coords[-1]) < tolerance and abs(y_max - y_coords[-1]) < tolerance:
            y_slice = (ny - 1, ny) # Last Y plane (top boundary)
            adjacent_slice = (ny - 2, ny - 1)
        elif abs(z_min - z_coords[0]) < tolerance and abs(z_max - z_coords[0]) < tolerance:
            z_slice = (0, 1) # First Z plane (front boundary)
            adjacent_slice = (1, 2)
        elif abs(z_min - z_coords[-1]) < tolerance and abs(z_max - z_coords[-1]) < tolerance:
            z_slice = (nz - 1, nz) # Last Z plane (back boundary)
            adjacent_slice = (nz - 2, nz - 1)
        elif y_pos is not None: # Specific for an internal 2D plane in Y
            y_idx = np.where(np.abs(y_coords - y_pos) < tolerance)[0]
            if len(y_idx) > 0:
                y_slice = (y_idx[0], y_idx[0] + 1)
                # For internal boundaries, adjacent_slice logic is more complex, might need two.
                # For simplicity, we'll assume external faces for Neumann for now.
                print(f"Warning: y_pos {y_pos} for boundary '{bc_name}' is an internal boundary. "
                      "Neumann/Pressure Outlet handling for internal faces is not fully supported with current simple slicing. "
                      "Assuming Dirichlet if applicable.")
            else:
                print(f"Warning: y_pos {y_pos} not found in y_coords. Skipping boundary {bc_name}.")
                continue
        else:
            print(f"Warning: Boundary condition '{bc_name}' cannot be fully represented by simple slices. "
                  "Only external faces are currently fully supported. Skipping.")
            continue


        # If a slice is None, it means it spans the full dimension
        final_x_slice = slice(x_slice[0], x_slice[1]) if x_slice else slice(None)
        final_y_slice = slice(y_slice[0], y_slice[1]) if y_slice else slice(None)
        final_z_slice = slice(z_slice[0], z_slice[1]) if z_slice else slice(None)

        # Determine which dimension the boundary is on for Neumann purposes
        # And determine the slice for the interior cell adjacent to the boundary
        boundary_dim = None # 0 for X, 1 for Y, 2 for Z
        if x_slice and (x_slice[0] == 0 or x_slice[0] == nx - 1):
            boundary_dim = 0
            if x_slice[0] == 0: interior_neighbor_slice = (slice(1,2), slice(None), slice(None))
            else: interior_neighbor_slice = (slice(nx-2,nx-1), slice(None), slice(None))
        elif y_slice and (y_slice[0] == 0 or y_slice[0] == ny - 1):
            boundary_dim = 1
            if y_slice[0] == 0: interior_neighbor_slice = (slice(None), slice(1,2), slice(None))
            else: interior_neighbor_slice = (slice(None), slice(ny-2,ny-1), slice(None))
        elif z_slice and (z_slice[0] == 0 or z_slice[0] == nz - 1):
            boundary_dim = 2
            if z_slice[0] == 0: interior_neighbor_slice = (slice(None), slice(None), slice(1,2))
            else: interior_neighbor_slice = (slice(None), slice(None), slice(nz-2,nz-1))
        else:
            interior_neighbor_slice = (slice(None), slice(None), slice(None)) # Not an axis-aligned external boundary for simple Neumann


        processed_bcs[bc_name] = {
            "type": bc_type,
            "slices": (final_x_slice, final_y_slice, final_z_slice),
            "velocity": np.array(bc_data.get("velocity", [0.0, 0.0, 0.0])), # Default to 0 velocity
            "pressure": bc_data.get("pressure", 0.0), # Default to 0 pressure
            "apply_to": bc_data.get("apply_to", ["velocity", "pressure"]), # What fields to apply to
            "boundary_dim": boundary_dim, # Which dimension the boundary is on
            "interior_neighbor_slice": interior_neighbor_slice # Slice of the adjacent interior cell(s)
        }

    return processed_bcs


def apply_boundary_conditions(velocity_field, pressure_field, fluid_properties, mesh_info, is_tentative_step):
    """
    Applies boundary conditions to the velocity and pressure fields using NumPy slicing.
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
        x_slice, y_slice, z_slice = bc["slices"]
        target_velocity = bc["velocity"]
        target_pressure = bc["pressure"]
        apply_to_fields = bc["apply_to"]
        boundary_dim = bc["boundary_dim"]
        interior_neighbor_slice = bc["interior_neighbor_slice"]

        # Construct the full slice tuple for the boundary face
        face_slice = (x_slice, y_slice, z_slice)

        # Dirichlet (Fixed Value) Boundary Conditions
        if bc_type == "dirichlet":
            if "velocity" in apply_to_fields:
                velocity_field[face_slice] = target_velocity
            if "pressure" in apply_to_fields and not is_tentative_step:
                # Apply pressure Dirichlet only for final pressure, not tentative velocity step
                pressure_field[face_slice] = target_pressure

        # Neumann (Fixed Gradient/Outflow) Boundary Conditions - Zero Gradient
        elif bc_type == "neumann":
            # For zero-gradient Neumann, set boundary cell value equal to adjacent interior cell value.
            # This applies to velocity and potentially pressure (outflow).

            # Prepare slice for adjacent interior cells
            if boundary_dim == 0: # X-boundary
                adj_x_slice, adj_y_slice, adj_z_slice = interior_neighbor_slice
            elif boundary_dim == 1: # Y-boundary
                adj_x_slice, adj_y_slice, adj_z_slice = interior_neighbor_slice
            elif boundary_dim == 2: # Z-boundary
                adj_x_slice, adj_y_slice, adj_z_slice = interior_neighbor_slice
            else:
                # This case implies a non-axis-aligned external boundary not handled by simple slices
                # or an internal boundary. Skip for now.
                print(f"Warning: Neumann BC '{bc_name}' on non-standard boundary or internal face. Skipping velocity/pressure application.")
                continue

            adj_cell_slice = (adj_x_slice, adj_y_slice, adj_z_slice)


            if "velocity" in apply_to_fields:
                # Make sure the interior slice covers the same (Y,Z) or (X,Z) or (X,Y) extent as the boundary slice
                # For example, if x_slice is (0,1), adj_x_slice is (1,2)
                # velocity_field[0,:,:,:] = velocity_field[1,:,:,:]
                velocity_field[face_slice] = velocity_field[adj_cell_slice]

            if "pressure" in apply_to_fields and not is_tentative_step:
                pressure_field[face_slice] = pressure_field[adj_cell_slice]


        # Pressure Outlet Boundary Condition (often combined with zero-gradient velocity)
        elif bc_type == "pressure_outlet":
            if "pressure" in apply_to_fields and not is_tentative_step:
                pressure_field[face_slice] = target_pressure

            # For velocity, usually zero-gradient outflow is applied, similar to Neumann
            # Using the same logic as Neumann for velocity at pressure outlets
            if "velocity" in apply_to_fields:
                 # Prepare slice for adjacent interior cells, same logic as Neumann
                if boundary_dim == 0: # X-boundary
                    adj_x_slice, adj_y_slice, adj_z_slice = interior_neighbor_slice
                elif boundary_dim == 1: # Y-boundary
                    adj_x_slice, adj_y_slice, adj_z_slice = interior_neighbor_slice
                elif boundary_dim == 2: # Z-boundary
                    adj_x_slice, adj_y_slice, adj_z_slice = interior_neighbor_slice
                else:
                    print(f"Warning: Pressure Outlet BC '{bc_name}' on non-standard boundary or internal face for velocity. Skipping velocity application.")
                    continue

                adj_cell_slice = (adj_x_slice, adj_y_slice, adj_z_slice)
                velocity_field[face_slice] = velocity_field[adj_cell_slice]

        else:
            print(f"Warning: Unknown boundary condition type '{bc_type}' for '{bc_name}'. Skipping.")