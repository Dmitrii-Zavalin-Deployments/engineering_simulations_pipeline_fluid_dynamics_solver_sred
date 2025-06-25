# src/physics/boundary_conditions_applicator.py

import numpy as np
import sys

# Assume a small tolerance for floating point comparisons
TOLERANCE = 1e-6

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
        return velocity_field, pressure_field
    if not isinstance(pressure_field, np.ndarray) or pressure_field.dtype != np.float64:
        print(f"ERROR (apply_boundary_conditions): pressure_field is not a float64 numpy array or is None. Type: {type(pressure_field)}, Dtype: {getattr(pressure_field, 'dtype', 'N/A')}", file=sys.stderr)
        return velocity_field, pressure_field # Return existing, potentially invalid fields

    print(f"DEBUG (apply_boundary_conditions): velocity_field shape: {velocity_field.shape}, pressure_field shape: {pressure_field.shape}")

    processed_bcs = mesh_info.get('boundary_conditions')
    if processed_bcs is None:
        print("ERROR (apply_boundary_conditions): 'boundary_conditions' not found in mesh_info. This is critical.", file=sys.stderr)
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

        # --- FIX: Changed .size == 0 to len() == 0 for robustness ---
        if len(cell_indices) == 0:
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