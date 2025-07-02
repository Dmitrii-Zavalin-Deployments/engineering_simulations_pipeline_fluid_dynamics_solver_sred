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
    print(f"DEBUG: apply_boundary_conditions called. is_tentative_step={is_tentative_step}")

    if not isinstance(velocity_field, np.ndarray) or velocity_field.dtype != np.float64:
        print(f"ERROR: velocity_field invalid. Type: {type(velocity_field)}", file=sys.stderr)
        return velocity_field, pressure_field
    if not isinstance(pressure_field, np.ndarray) or pressure_field.dtype != np.float64:
        print(f"ERROR: pressure_field invalid. Type: {type(pressure_field)}", file=sys.stderr)
        return velocity_field, pressure_field

    processed_bcs = mesh_info.get("boundary_conditions", {})
    if not processed_bcs:
        print("WARNING: No boundary_conditions found in mesh_info. Skipping BC application.", file=sys.stderr)
        return velocity_field, pressure_field

    for bc_name, bc in processed_bcs.items():
        if "cell_indices" not in bc or "ghost_indices" not in bc:
            print(f"WARNING: BC '{bc_name}' is missing indices. Skipping. Was pre-processing successful?", file=sys.stderr)
            continue

        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])
        cell_indices = np.array(bc["cell_indices"])
        ghost_indices = np.array(bc["ghost_indices"])

        print(f"[BC DEBUG] Processing '{bc_name}': type='{bc_type}', apply_to={apply_to_fields}")

        if bc_type == "dirichlet":
            if "velocity" in apply_to_fields:
                if ghost_indices.size > 0:
                    target_velocity = bc.get("velocity", [0.0, 0.0, 0.0])
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = target_velocity
                    print(f"    -> Applied Dirichlet velocity {target_velocity} to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No ghost cells for velocity BC '{bc_name}'.")

            if "pressure" in apply_to_fields and not is_tentative_step:
                if ghost_indices.size > 0:
                    target_pressure = bc.get("pressure", 0.0)
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = target_pressure
                    print(f"    -> Applied Dirichlet pressure {target_pressure} to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No ghost cells for pressure BC '{bc_name}'.")

        elif bc_type == "neumann":
            if "velocity" in apply_to_fields:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    velocity_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2], :] = \
                        velocity_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :]
                    print(f"    -> Applied Neumann velocity (zero-gradient) to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No cells for Neumann velocity BC '{bc_name}'.")

            if "pressure" in apply_to_fields and not is_tentative_step:
                if cell_indices.size > 0 and ghost_indices.size > 0:
                    pressure_field[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = \
                        pressure_field[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]]
                    print(f"    -> Applied Neumann pressure (zero-gradient) to ghost cells for '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No cells for Neumann pressure BC '{bc_name}'.")

        elif bc_type == "outflow":
            if "pressure" in apply_to_fields and not is_tentative_step:
                if ghost_indices.size > 0:
                    target_pressure = bc.get("pressure", 0.0)
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
    DEPRECATED: This function is replaced by apply_boundary_conditions and should not be used.
    """
    print(f"WARNING: apply_ghost_cells() is deprecated. Use apply_boundary_conditions() instead.")
    gx = 1
    field[0, :, :] = field[1, :, :]
    field[-1, :, :] = field[-2, :, :]
    field[:, 0, :] = field[:, 1, :]
    field[:, -1, :] = field[:, -2, :]
    field[:, :, 0] = field[:, :, 1]
    field[:, :, -1] = field[:, :, -2]



