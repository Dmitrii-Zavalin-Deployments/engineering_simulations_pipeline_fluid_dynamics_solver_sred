# src/physics/boundary_conditions_applicator.py

import numpy as np
import sys
from typing import Tuple

def apply_boundary_conditions(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    fluid_properties: dict, # Not directly used in this function, but kept for signature consistency
    mesh_info: dict,
    is_tentative_step: bool,
    step_number: int,          # Added for conditional logging
    output_frequency_steps: int # Added for conditional logging
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies boundary conditions to the velocity and pressure fields by modifying their ghost cells.

    Args:
        velocity_field (np.ndarray): The velocity field (u, v, w components) with ghost cells.
                                     Shape: (nx+2, ny+2, nz+2, 3).
        pressure_field (np.ndarray): The pressure field with ghost cells.
                                     Shape: (nx+2, ny+2, nz+2).
        fluid_properties (dict): Dictionary containing fluid properties (e.g., 'density', 'viscosity').
                                 Not directly used in this function but passed for consistency.
        mesh_info (dict): Dictionary containing grid and boundary condition information,
                          including 'boundary_conditions' which holds pre-processed indices.
        is_tentative_step (bool): True if applying BCs after the tentative velocity step (u*),
                                  False if applying after the pressure correction step (u_new, p_new).
        step_number (int): Current simulation step number, used for conditional logging.
        output_frequency_steps (int): Frequency for printing debug output, used for conditional logging.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The velocity and pressure fields with updated ghost cells.
    """
    if step_number % output_frequency_steps == 0:
        print(f"DEBUG: apply_boundary_conditions called. is_tentative_step={is_tentative_step}")

    # Input validation: Ensure fields are NumPy arrays of float64 type
    if not isinstance(velocity_field, np.ndarray) or velocity_field.dtype != np.float64:
        if step_number % output_frequency_steps == 0:
            print(f"ERROR: velocity_field invalid. Type: {type(velocity_field)}, Dtype: {velocity_field.dtype}", file=sys.stderr)
        return velocity_field, pressure_field
    if not isinstance(pressure_field, np.ndarray) or pressure_field.dtype != np.float64:
        if step_number % output_frequency_steps == 0:
            print(f"ERROR: pressure_field invalid. Type: {type(pressure_field)}, Dtype: {pressure_field.dtype}", file=sys.stderr)
        return velocity_field, pressure_field

    # Retrieve pre-processed boundary conditions from mesh_info
    processed_bcs = mesh_info.get("boundary_conditions", {})
    if not processed_bcs:
        if step_number % output_frequency_steps == 0:
            print("WARNING: No boundary_conditions found in mesh_info. Skipping BC application.", file=sys.stderr)
        return velocity_field, pressure_field

    # Iterate through each defined boundary condition
    for bc_name, bc in processed_bcs.items():
        # Ensure necessary indices are present (should be from pre-processing)
        if "cell_indices" not in bc or "ghost_indices" not in bc:
            if step_number % output_frequency_steps == 0:
                print(f"WARNING: BC '{bc_name}' is missing indices. Skipping. Was pre-processing successful?", file=sys.stderr)
            continue

        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])
        # Convert index lists to NumPy arrays for efficient indexing
        cell_indices = np.array(bc["cell_indices"], dtype=int)
        ghost_indices = np.array(bc["ghost_indices"], dtype=int)

        if step_number % output_frequency_steps == 0:
            print(f"[BC DEBUG] Processing '{bc_name}': type='{bc_type}', apply_to={apply_to_fields}")

        if bc_type == "dirichlet":
            # Apply Dirichlet velocity boundary conditions (fixed velocity)
            if "velocity" in apply_to_fields:
                if ghost_indices.size > 0:
                    target_velocity = bc.get("velocity", [0.0, 0.0, 0.0])
                    # Set ghost cell velocity directly to the target velocity
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = target_velocity
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> Applied Dirichlet velocity {target_velocity} to ghost cells for '{bc_name}'.")
                else:
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> WARNING: No ghost cells found for Dirichlet velocity BC '{bc_name}'.")

            # Apply Dirichlet pressure boundary conditions (fixed pressure)
            # Pressure BCs are typically applied only after the pressure correction step (final step)
            if "pressure" in apply_to_fields and not is_tentative_step:
                if ghost_indices.size > 0:
                    target_pressure = bc.get("pressure", 0.0)
                    # Set ghost cell pressure directly to the target pressure
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = target_pressure
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> Applied Dirichlet pressure {target_pressure} to ghost cells for '{bc_name}'.")
                else:
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> WARNING: No ghost cells found for Dirichlet pressure BC '{bc_name}'.")

        elif bc_type == "neumann":
            # Apply Neumann velocity boundary conditions (zero-gradient, i.e., ghost = adjacent interior)
            if "velocity" in apply_to_fields:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    # Copy velocity from adjacent interior cells to ghost cells
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = \
                        velocity_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :]
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> Applied Neumann velocity (zero-gradient) to ghost cells for '{bc_name}'.")
                else:
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> WARNING: No valid cell/ghost pairs for Neumann velocity BC '{bc_name}'.")

            # Apply Neumann pressure boundary conditions (zero-gradient)
            # Pressure BCs are typically applied only after the pressure correction step (final step)
            if "pressure" in apply_to_fields and not is_tentative_step:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    # Copy pressure from adjacent interior cells to ghost cells
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = \
                        pressure_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]]
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> Applied Neumann pressure (zero-gradient) to ghost cells for '{bc_name}'.")
                else:
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> WARNING: No valid cell/ghost pairs for Neumann pressure BC '{bc_name}'.")

        elif bc_type == "outflow":
            # Outflow boundary conditions typically involve zero-gradient for velocity
            # and a fixed pressure (often zero) or zero-gradient for pressure.
            # Current implementation uses Dirichlet pressure and Neumann velocity.
            if "pressure" in apply_to_fields and not is_tentative_step:
                if ghost_indices.size > 0:
                    target_pressure = bc.get("pressure", 0.0)
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = target_pressure
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> Applied Dirichlet pressure {target_pressure} for outflow BC '{bc_name}'.")
                else:
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> WARNING: No ghost cells found for outflow pressure BC '{bc_name}'.")

            if "velocity" in apply_to_fields:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = \
                        velocity_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :]
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> Applied Neumann velocity (zero-gradient) for outflow BC '{bc_name}'.")
                else:
                    if step_number % output_frequency_steps == 0:
                        print(f"    -> WARNING: No valid cell/ghost pairs for outflow velocity BC '{bc_name}'.")

        else:
            if step_number % output_frequency_steps == 0:
                print(f"WARNING: Unknown BC type '{bc_type}' for '{bc_name}'. Skipping.", file=sys.stderr)

    if step_number % output_frequency_steps == 0:
        print("DEBUG: apply_boundary_conditions completed.")
    return velocity_field, pressure_field


def apply_ghost_cells(field: np.ndarray, field_name: str):
    """
    DEPRECATED: This function is replaced by apply_boundary_conditions and should not be used.
    It performs a simple zero-gradient (Neumann) boundary condition application by copying
    values from adjacent interior cells to ghost cells.
    """
    print(f"WARNING: apply_ghost_cells() is deprecated. Use apply_boundary_conditions() instead.")
    # Apply zero-gradient to all faces
    field[0, :, :] = field[1, :, :]
    field[-1, :, :] = field[-2, :, :]
    field[:, 0, :] = field[:, 1, :]
    field[:, -1, :] = field[:, -2, :]
    field[:, :, 0] = field[:, :, 1]
    field[:, :, -1] = field[:, :, -2]



