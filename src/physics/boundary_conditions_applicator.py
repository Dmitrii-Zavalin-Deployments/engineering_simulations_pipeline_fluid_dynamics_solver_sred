# src/physics/boundary_conditions_applicator.py

import numpy as np
import sys
from typing import Tuple

TOLERANCE = 1e-6

def apply_boundary_conditions(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    fluid_properties: dict,
    mesh_info: dict,
    is_tentative_step: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies boundary conditions to the velocity and pressure fields using NumPy indexing.
    Modifies input fields in-place. This function is called at two key stages in the solver:
    1. After the advection-diffusion step (is_tentative_step=True) to enforce velocity BCs on u*.
    2. After the pressure correction step (is_tentative_step=False) to enforce all BCs on the final fields.

    Args:
        velocity_field (np.ndarray): Velocity field (nx, ny, nz, 3).
        pressure_field (np.ndarray): Pressure field (nx, ny, nz).
        fluid_properties (dict): Dictionary with fluid properties.
        mesh_info (dict): Grid info including 'grid_shape' and 'boundary_conditions'.
        is_tentative_step (bool): True for u*, False for uⁿ⁺¹ or pressure.
    """
    print(f"DEBUG: apply_boundary_conditions called. is_tentative_step={is_tentative_step}")

    # --- Sanity Checks ---
    if not isinstance(velocity_field, np.ndarray) or velocity_field.dtype != np.float64:
        print(f"ERROR: velocity_field invalid. Type: {type(velocity_field)}, Dtype: {getattr(velocity_field, 'dtype', 'N/A')}", file=sys.stderr)
        return velocity_field, pressure_field
    if not isinstance(pressure_field, np.ndarray) or pressure_field.dtype != np.float64:
        print(f"ERROR: pressure_field invalid. Type: {type(pressure_field)}, Dtype: {getattr(pressure_field, 'dtype', 'N/A')}", file=sys.stderr)
        return velocity_field, pressure_field

    processed_bcs = mesh_info.get('boundary_conditions', {})
    if not processed_bcs:
        print("WARNING: No boundary_conditions found in mesh_info. Skipping BC application.", file=sys.stderr)
        return velocity_field, pressure_field

    for bc_name, bc in processed_bcs.items():
        if "cell_indices" not in bc or "ghost_indices" not in bc:
            print(f"WARNING: BC '{bc_name}' is missing indices. Skipping. Was pre-processing successful?", file=sys.stderr)
            continue
            
        bc_data = bc.get("data", {})
        bc_type = bc_data.get("type")
        apply_to_fields = bc_data.get("apply_to", [])
            
        # Get the indices for the boundary cells and the ghost cells
        cell_indices = np.array(bc["cell_indices"])
        ghost_indices = np.array(bc["ghost_indices"])
            
        print(f"[BC DEBUG] Processing '{bc_name}': type='{bc_type}', apply_to={apply_to_fields}")
            
        # --- 1. Handle Dirichlet BCs (Fixed values) ---
        if bc_type == "dirichlet":
            # Apply Dirichlet Velocity BCs (e.g., No-slip walls, fixed inlet velocity)
            if "velocity" in apply_to_fields:
                if ghost_indices.size > 0:
                    target_velocity = bc_data.get("velocity", [0.0, 0.0, 0.0])
                    # Apply the velocity to the ghost cells
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = target_velocity
                    print(f"    -> Applied Dirichlet velocity {target_velocity} to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No ghost cells identified for velocity BC '{bc_name}'.")

            # Apply Dirichlet Pressure BCs (Fixed pressure inlet/outlet)
            # This is applied to the final pressure field after the solve.
            if "pressure" in apply_to_fields and not is_tentative_step:
                if ghost_indices.size > 0:
                    target_pressure = bc_data.get("pressure", 0.0)
                    # Apply the pressure to the ghost cells
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = target_pressure
                    print(f"    -> Applied Dirichlet pressure {target_pressure} to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No ghost cells identified for pressure BC '{bc_name}'.")
            
        # --- 2. Handle Neumann BCs (Zero-gradient) ---
        elif bc_type == "neumann":
            # Apply Neumann Velocity BCs (e.g., zero-gradient outlet)
            if "velocity" in apply_to_fields:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    # Copy the velocity from the boundary cells to the ghost cells.
                    # This implies a zero-gradient (Neumann) condition.
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = \
                        velocity_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :]
                    print(f"    -> Applied Neumann velocity (zero-gradient) to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No boundary/ghost cells found for Neumann velocity BC '{bc_name}'.")

            # Apply Neumann Pressure BCs (e.g., zero-gradient pressure)
            if "pressure" in apply_to_fields and not is_tentative_step:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    # Copy pressure from boundary cells to ghost cells
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = \
                        pressure_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]]
                    print(f"    -> Applied Neumann pressure (zero-gradient) to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No boundary/ghost cells found for Neumann pressure BC '{bc_name}'.")

        # --- 3. Handle a combined 'outflow' BC type (Optional, based on your JSON) ---
        # Note: Your JSON uses 'dirichlet' for both inlet and outlet pressure,
        # so this 'outflow' type may not be used directly, but the logic is fine.
        elif bc_type == "outflow":
            if "pressure" in apply_to_fields and not is_tentative_step:
                if ghost_indices.size > 0:
                    target_pressure = bc_data.get("pressure", 0.0)
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = target_pressure
                    print(f"    -> Applied Dirichlet pressure {target_pressure} for outflow BC '{bc_name}'.")
            
            if "velocity" in apply_to_fields:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = \
                        velocity_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :]
                    print(f"    -> Applied Neumann velocity (zero-gradient) for outflow BC '{bc_name}'.")
                    
        else:
            print(f"WARNING: Unknown BC type '{bc_type}' for '{bc_name}'. Skipping.", file=sys.stderr)

    print("DEBUG: apply_boundary_conditions completed.")
    return velocity_field, pressure_field


def apply_ghost_cells(field: np.ndarray, field_name: str):
    """
    DEPRECATED: This function is replaced by the more robust apply_boundary_conditions.
    Kept for reference but should not be used in the main solver loop.
    """
    print(f"WARNING: apply_ghost_cells() is deprecated. Use apply_boundary_conditions() instead.")
    gx = 1 # single ghost layer
    field[0, :, :] = field[1, :, :]
    field[-1, :, :] = field[-2, :, :]
    field[:, 0, :] = field[:, 1, :]
    field[:, -1, :] = field[:, -2, :]
    field[:, :, 0] = field[:, :, 1]
    field[:, :, -1] = field[:, :, -2]


