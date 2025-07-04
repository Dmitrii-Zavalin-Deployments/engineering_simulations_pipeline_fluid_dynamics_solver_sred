# src/physics/boundary_conditions_applicator.py

import numpy as np

def apply_boundary_conditions(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    fluid_properties: dict,
    mesh_info: dict,
    is_tentative_step: bool,
    should_log_verbose: bool = False # Added verbose logging flag
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies boundary conditions to the velocity and pressure fields.

    Args:
        velocity_field (np.ndarray): The velocity field with ghost cells (nx+2, ny+2, nz+2, 3).
        pressure_field (np.ndarray): The pressure field with ghost cells (nx+2, ny+2, nz+2).
        fluid_properties (dict): Dictionary containing fluid properties (e.g., 'U_LID').
        mesh_info (dict): Dictionary containing mesh dimensions 'nx', 'ny', 'nz'.
        is_tentative_step (bool): True if applying BCs after advection/diffusion (u*),
                                  False if applying final BCs after pressure correction.
        should_log_verbose (bool): If True, print detailed debug logs.

    Returns:
        tuple[np.ndarray, np.ndarray]: The velocity and pressure fields with BCs applied.
    """
    nx, ny, nz = mesh_info['nx'], mesh_info['ny'], mesh_info['nz']
    U_LID = fluid_properties.get('U_LID', 0.0) # Lid velocity for lid-driven cavity

    # --- Apply Velocity Boundary Conditions ---
    # Velocity ghost cells are set to enforce desired conditions at physical boundaries.

    # 1. Walls (No-slip: u=0, v=0, w=0)
    # Bottom wall (z=0)
    velocity_field[:, :, 0, :] = -velocity_field[:, :, 1, :] # Reflective for no-slip
    # Top wall (z=max) - This is usually the lid for lid-driven cavity
    if U_LID != 0.0:
        # For lid-driven cavity, top wall has non-zero tangential velocity
        # Assuming lid moves in x-direction
        velocity_field[:, :, nz + 1, 0] = 2 * U_LID - velocity_field[:, :, nz, 0] # U-component
        velocity_field[:, :, nz + 1, 1] = -velocity_field[:, :, nz, 1] # V-component (0)
        velocity_field[:, :, nz + 1, 2] = -velocity_field[:, :, nz, 2] # W-component (0)
    else:
        # Standard no-slip top wall
        velocity_field[:, :, nz + 1, :] = -velocity_field[:, :, nz, :]


    # Front wall (y=0) and Back wall (y=max)
    velocity_field[:, 0, :, :] = -velocity_field[:, 1, :, :]
    velocity_field[:, ny + 1, :, :] = -velocity_field[:, ny, :, :]

    # Left wall (x=0) and Right wall (x=max)
    velocity_field[0, :, :, :] = -velocity_field[1, :, :, :]
    velocity_field[nx + 1, :, :, :] = -velocity_field[nx, :, :, :]

    # --- Apply Pressure Boundary Conditions ---
    # Pressure ghost cells typically derived from interior values (Neumann conditions, dP/dn = 0)
    # or fixed (Dirichlet). For incompressible flow, pressure often has Neumann BCs at walls.

    # If it's the tentative step, pressure ghost cells are typically set to match
    # the adjacent interior cells (Neumann/dP/dn=0).
    # After pressure correction, pressure values at boundaries are also updated.

    # Neumann condition (dP/dn = 0) at all boundaries for pressure
    # This means pressure in ghost cell equals pressure in adjacent interior cell.
    pressure_field[0, :, :] = pressure_field[1, :, :]      # Left
    pressure_field[nx + 1, :, :] = pressure_field[nx, :, :]  # Right
    pressure_field[:, 0, :] = pressure_field[:, 1, :]      # Front
    pressure_field[:, ny + 1, :] = pressure_field[:, ny, :]  # Back
    pressure_field[:, :, 0] = pressure_field[:, :, 1]      # Bottom
    pressure_field[:, :, nz + 1] = pressure_field[:, :, nz]  # Top

    # An alternative for pressure (common for lid-driven cavity) is to fix
    # pressure at one point to remove the arbitrary constant.
    # We will assume that pressure_field is initially normalized or adjusted
    # elsewhere, or that the Neumann conditions are sufficient.
    # For now, let's keep pressure ghost cells simply equal to interior.

    # Ensure no NaNs or Infs are introduced, though they shouldn't be by these operations
    if np.isnan(velocity_field).any() or np.isinf(velocity_field).any():
        print("❌ Warning: Invalid values detected in velocity_field after BCs. Clamping to zero.")
        velocity_field = np.nan_to_num(velocity_field, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(pressure_field).any() or np.isinf(pressure_field).any():
        print("❌ Warning: Invalid values detected in pressure_field after BCs. Clamping to zero.")
        pressure_field = np.nan_to_num(pressure_field, nan=0.0, posinf=0.0, neginf=0.0)


    if should_log_verbose:
        step_type = "tentative" if is_tentative_step else "final"
        print(f"    - Boundary Conditions applied for {step_type} step.")
        # Add more specific checks if needed, e.g., max velocity at lid
        if U_LID != 0.0 and step_type == "final":
             # Check u-velocity at the top lid boundary (z=nz+1 ghost cell, corresponding to nz physical)
             # The average of the ghost cell and physical cell at the boundary should approximate U_LID
             actual_lid_u_avg = np.mean(velocity_field[1:-1, 1:-1, nz, 0]) # Average u at physical top layer
             print(f"        • Avg U at physical top (z={nz}): {actual_lid_u_avg:.4e} (Target U_LID: {U_LID:.4e})")


    return velocity_field, pressure_field



