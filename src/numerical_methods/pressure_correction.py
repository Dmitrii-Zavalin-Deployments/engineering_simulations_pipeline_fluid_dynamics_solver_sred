import numpy as np
import sys

def apply_pressure_correction(
    tentative_velocity_field: np.ndarray,
    current_pressure_field: np.ndarray,
    phi: np.ndarray, # This phi is the potential from Poisson solver
    dt: float,
    density: float,
    mesh_info: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the pressure correction to the tentative velocity field and updates the pressure.

    ∇P_new = ∇P_old + (ρ/Δt)∇φ
    u_new = u_star - (Δt/ρ)∇φ

    Args:
        tentative_velocity_field (np.ndarray): The velocity field after advection and diffusion (u_star).
                                               Shape (nx_total, ny_total, nz_total, 3)
        current_pressure_field (np.ndarray): The pressure field from the previous time step.
                                               Shape (nx_total, ny_total, nz_total)
        phi (np.ndarray): The pressure correction potential obtained from the Poisson solver.
                          This MUST be of shape (nx_total, ny_total, nz_total)
                          matching `current_pressure_field`.
        dt (float): Time step size.
        density (float): Fluid density.
        mesh_info (dict): Dictionary with mesh details including 'grid_shape', 'dx', 'dy', 'dz',
                          and boundary condition info.

    Returns:
        tuple: (updated_velocity_field, updated_pressure_field,
                grad_phi_x, grad_phi_y, grad_phi_z, divergence_after_correction_field)
    """
    print("\n[DEBUG pressure_correction.py] Inside apply_pressure_correction:")
    print(f"[DEBUG pressure_correction.py] tentative_velocity_field shape: {tentative_velocity_field.shape}")
    print(f"[DEBUG pressure_correction.py] current_pressure_field shape: {current_pressure_field.shape}")
    print(f"[DEBUG pressure_correction.py] phi shape: {phi.shape}")

    nx_total, ny_total, nz_total = current_pressure_field.shape
    print(f"[DEBUG pressure_correction.py] Derived nx_total, ny_total, nz_total from current_pressure_field.shape: {nx_total}, {ny_total}, {nz_total}")

    # Sanity check for phi's shape. This is where the mismatch was.
    if phi.shape != (nx_total, ny_total, nz_total):
        print(f"ERROR: phi shape {phi.shape} does not match expected total grid shape {(nx_total, ny_total, nz_total)}.", file=sys.stderr)
        print("Please ensure solve_poisson_for_phi returns phi on the full grid, including ghost cells.", file=sys.stderr)
        raise ValueError("Phi shape mismatch in apply_pressure_correction.")

    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Initialize updated velocity and pressure fields
    updated_velocity_field = np.copy(tentative_velocity_field)
    updated_pressure = np.copy(current_pressure_field)

    # Define interior dimensions (excluding 1 layer of ghost cells)
    nx_interior = nx_total - 2
    ny_interior = ny_total - 2
    nz_interior = nz_total - 2

    print(f"[DEBUG pressure_correction.py] Derived nx_interior, ny_interior, nz_interior: {nx_interior}, {ny_interior}, {nz_interior}")

    # Slice for cell centers (excluding ghost cells) for pressure update
    # This refers to the actual computational domain, not including the ghost cells.
    interior_slice = (slice(1, nx_total - 1), slice(1, ny_total - 1), slice(1, nz_total - 1))
    print(f"[DEBUG pressure_correction.py] interior_slice: {interior_slice}")
    
    # 1. Calculate the gradient of phi at cell faces for velocity correction
    # These gradients are computed using central differences of phi.
    # They should align with the staggered grid locations of the velocity components.
    
    # Gradient of phi in x-direction (for u_x at i+0.5 faces)
    # u_x velocity component is defined on faces between cell i and i+1.
    # The array `updated_velocity_field[...,0]` contains u-velocities from index 0 to nx_total-1.
    # The *interior* u-velocities are at indices 1 to nx_total-1 (i.e., `updated_velocity_field[1:nx_total, :, :, 0]`).
    # This means the gradient should have `nx_total-1` elements in the x-direction.
    grad_phi_x_at_faces = (
        phi[1:nx_total, interior_slice[1], interior_slice[2]] -
        phi[0:nx_total-1, interior_slice[1], interior_slice[2]]
    ) / dx
    # Expected shape: (nx_total-1, ny_interior, nz_interior)
    # For nx_total=23, ny_total=23, nz_total=7 -> (22, 21, 5)
    print(f"[DEBUG pressure_correction.py] grad_phi_x_at_faces pre-check: {grad_phi_x_at_faces.shape}")


    # Gradient of phi in y-direction (for u_y at j+0.5 faces)
    # v_y velocity component is defined on faces between cell j and j+1.
    # Similar logic, it should have `ny_total-1` elements in the y-direction.
    grad_phi_y_at_faces = (
        phi[interior_slice[0], 1:ny_total, interior_slice[2]] -
        phi[interior_slice[0], 0:ny_total-1, interior_slice[2]]
    ) / dy
    # Expected shape: (nx_interior, ny_total-1, nz_interior)
    # For nx_total=23, ny_total=23, nz_total=7 -> (21, 22, 5)
    print(f"[DEBUG pressure_correction.py] grad_phi_y_at_faces pre-check: {grad_phi_y_at_faces.shape}")


    # Gradient of phi in z-direction (for u_z at k+0.5 faces)
    # w_z velocity component is defined on faces between cell k and k+1.
    # Similar logic, it should have `nz_total-1` elements in the z-direction.
    grad_phi_z_at_faces = (
        phi[interior_slice[0], interior_slice[1], 1:nz_total] -
        phi[interior_slice[0], interior_slice[1], 0:nz_total-1]
    ) / dz
    # Expected shape: (nx_interior, ny_interior, nz_total-1)
    # For nx_total=23, ny_total=23, nz_total=7 -> (21, 21, 6)
    print(f"[DEBUG pressure_correction.py] grad_phi_z_at_faces pre-check: {grad_phi_z_at_faces.shape}")


    # 2. Update velocity field (u_new = u_star - (Δt/ρ)∇φ)
    # Apply correction to X-velocity (u_x)
    # The u-velocity components that are updated are those at indices `1` to `nx_total-1`.
    updated_velocity_field[1:nx_total, interior_slice[1], interior_slice[2], 0] -= (dt / density) * grad_phi_x_at_faces
    
    # Apply correction to Y-velocity (u_y)
    # The v-velocity components that are updated are those at indices `1` to `ny_total-1`.
    updated_velocity_field[interior_slice[0], 1:ny_total, interior_slice[2], 1] -= (dt / density) * grad_phi_y_at_faces

    # Apply correction to Z-velocity (u_z)
    # The w-velocity components that are updated are those at indices `1` to `nz_total-1`.
    updated_velocity_field[interior_slice[0], interior_slice[1], 1:nz_total, 2] -= (dt / density) * grad_phi_z_at_faces

    # 3. Update pressure (P_new = P_old + ρφ)
    # This update is for the interior pressure cells.
    updated_pressure[interior_slice] += density * phi[interior_slice]

    # 4. Re-calculate divergence after correction for debug/validation
    # Compute divergence of the corrected velocity field for interior cells
    divergence_after_correction_field = np.zeros_like(current_pressure_field) # Same shape as pressure

    # For interior cells (1 to N-2)
    # ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
    # ∂u/∂x at cell center (i,j,k) approximated using central difference of u_x at faces: (u_x[i+1/2,j,k] - u_x[i-1/2,j,k]) / dx
    # This corresponds to:
    # (updated_velocity_field[i+1,j,k,0] - updated_velocity_field[i,j,k,0]) / dx for interior i.
    # The updated_velocity_field at `i` is the face velocity just before the cell center `i`.
    divergence_after_correction_field[interior_slice] = (
        (updated_velocity_field[interior_slice[0].start + 1 : interior_slice[0].stop + 1, interior_slice[1], interior_slice[2], 0] -
         updated_velocity_field[interior_slice[0].start : interior_slice[0].stop, interior_slice[1], interior_slice[2], 0]) / dx + # u_x
        
        (updated_velocity_field[interior_slice[0], interior_slice[1].start + 1 : interior_slice[1].stop + 1, interior_slice[2], 1] -
         updated_velocity_field[interior_slice[0], interior_slice[1].start : interior_slice[1].stop, interior_slice[2], 1]) / dy + # u_y
        
        (updated_velocity_field[interior_slice[0], interior_slice[1], interior_slice[2].start + 1 : interior_slice[2].stop + 1, 2] -
         updated_velocity_field[interior_slice[0], interior_slice[1], interior_slice[2].start : interior_slice[2].stop, 2]) / dz     # u_z
    )

    return updated_velocity_field, updated_pressure, \
           grad_phi_x_at_faces, grad_phi_y_at_faces, grad_phi_z_at_faces, \
           divergence_after_correction_field



