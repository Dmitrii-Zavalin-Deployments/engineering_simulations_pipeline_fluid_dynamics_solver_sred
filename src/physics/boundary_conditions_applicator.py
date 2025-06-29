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
    Modifies input fields in-place.

    Args:
        velocity_field (np.ndarray): Velocity field (nx, ny, nz, 3).
        pressure_field (np.ndarray): Pressure field (nx, ny, nz).
        fluid_properties (dict): Dictionary with fluid properties.
        mesh_info (dict): Grid info including 'grid_shape' and 'boundary_conditions'.
        is_tentative_step (bool): True for u*, False for uⁿ⁺¹ or pressure.
    """
    print(f"DEBUG: apply_boundary_conditions called. is_tentative_step={is_tentative_step}")

    if not isinstance(velocity_field, np.ndarray) or velocity_field.dtype != np.float64:
        print(f"ERROR: velocity_field invalid. Type: {type(velocity_field)}, Dtype: {getattr(velocity_field, 'dtype', 'N/A')}", file=sys.stderr)
        return velocity_field, pressure_field
    if not isinstance(pressure_field, np.ndarray) or pressure_field.dtype != np.float64:
        print(f"ERROR: pressure_field invalid. Type: {type(pressure_field)}, Dtype: {getattr(pressure_field, 'dtype', 'N/A')}", file=sys.stderr)
        return velocity_field, pressure_field

    processed_bcs = mesh_info.get('boundary_conditions', {})
    if not processed_bcs:
        print("WARNING: No boundary_conditions found in mesh_info.", file=sys.stderr)
        return velocity_field, pressure_field

    nx, ny, nz = mesh_info['grid_shape']

    for bc_name, bc in processed_bcs.items():
        if "cell_indices" not in bc:
            raise ValueError(
                f"❌ Boundary condition '{bc_name}' is missing 'cell_indices'.\n"
                "Was the input preprocessed with 'identify_boundary_nodes()'?"
            )

        bc_type = bc.get("type")
        cell_indices = np.array(bc["cell_indices"])
        target_velocity = bc.get("velocity", [0.0, 0.0, 0.0])
        target_pressure = bc.get("pressure", 0.0)
        apply_to_fields = bc.get("apply_to") or []
        boundary_dim = bc.get("boundary_dim")
        offset = bc.get("interior_neighbor_offset", [0, 0, 0])

        if cell_indices.size == 0:
            print(f"WARNING: No cells for boundary '{bc_name}'. Skipping.", file=sys.stderr)
            continue

        if bc_type == "dirichlet":
            if "velocity" in apply_to_fields:
                max_indices = np.array(velocity_field.shape[:3])
                if np.any(np.max(cell_indices, axis=0) >= max_indices):
                    print(f"ERROR: Boundary '{bc_name}' has cell indices beyond grid shape {max_indices.tolist()}", file=sys.stderr)
                    raise IndexError(f"Invalid cell_indices for velocity in '{bc_name}' — indices exceed grid bounds")
                velocity_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_velocity
            if "pressure" in apply_to_fields and not is_tentative_step:
                if np.any(np.max(cell_indices, axis=0) >= pressure_field.shape):
                    print(f"ERROR: Pressure indices out of bounds for '{bc_name}'", file=sys.stderr)
                    raise IndexError(f"Invalid cell_indices for pressure in '{bc_name}' — indices exceed grid bounds")
                pressure_field[cell_indices[:,0], cell_indices[:,1], cell_indices[:,2]] = target_pressure

        elif bc_type in ["neumann", "pressure_outlet"]:
            neighbor = cell_indices + offset
            valid = (neighbor[:, 0] >= 0) & (neighbor[:, 0] < nx) & \
                    (neighbor[:, 1] >= 0) & (neighbor[:, 1] < ny) & \
                    (neighbor[:, 2] >= 0) & (neighbor[:, 2] < nz)

            inner = cell_indices[valid]
            neigh = neighbor[valid]

            if "velocity" in apply_to_fields:
                velocity_field[inner[:,0], inner[:,1], inner[:,2]] = velocity_field[neigh[:,0], neigh[:,1], neigh[:,2]]

            if bc_type == "pressure_outlet" and "pressure" in apply_to_fields and not is_tentative_step:
                pressure_field[inner[:,0], inner[:,1], inner[:,2]] = target_pressure
        else:
            print(f"WARNING: Unknown BC type '{bc_type}' for '{bc_name}'.", file=sys.stderr)

    return velocity_field, pressure_field


def apply_ghost_cells(field: np.ndarray, field_name: str):
    """
    Simple in-place ghost cell filler for test purposes.
    Copies interior slices to boundary ghost layers (periodic-style default).

    Args:
        field (np.ndarray): 3D array with ghost cells.
        field_name (str): For future extension (e.g., 'u', 'v', 'w').
    """
    gx = 1  # single ghost layer
    field[0, :, :] = field[1, :, :]
    field[-1, :, :] = field[-2, :, :]
    field[:, 0, :] = field[:, 1, :]
    field[:, -1, :] = field[:, -2, :]
    field[:, :, 0] = field[:, :, 1]
    field[:, :, -1] = field[:, :, -2]



