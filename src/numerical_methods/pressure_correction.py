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
                                               Shape (nx+2, ny+2, nz+2, 3)
        current_pressure_field (np.ndarray): The pressure field from the previous time step.
                                               Shape (nx+2, ny+2, nz+2)
        phi (np.ndarray): The pressure correction potential obtained from the Poisson solver.
                          This should be of shape (nx_total, ny_total, nz_total)
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
    cell_center_slice = (slice(1, nx_total - 1), slice(1, ny_total - 1), slice(1, nz_total - 1))
    print(f"[DEBUG pressure_correction.py] cell_center_slice: {cell_center_slice}")
    
    # 1. Calculate the gradient of phi at cell faces for velocity correction
    # Recall that phi is defined at cell centers.
    # Velocity components (u, v, w) are at cell faces:
    # u_x at (i+0.5, j, k)
    # u_y at (i, j+0.5, k)
    # u_z at (i, j, k+0.5)

    # For grad_phi_x, it should be calculated where u-velocity components live.
    # These are on the faces between (i) and (i+1) cell centers.
    # The u-velocity component at index (i,j,k) actually represents the velocity at face (i-0.5, j, k).
    # The *interior* u-velocities are at indices (1, :, :, 0) to (nx_total-1, :, :, 0).
    # So, we need (nx_total-1) values in the x-direction.

    # Gradient of phi in x-direction (for u_x at i+0.5 faces)
    # This calculation creates a gradient array that has 'nx_total - 1' elements in x.
    # It takes `phi[i+1, j, k] - phi[i, j, k]`.
    # The `j` and `k` indices remain for the interior cells.
    # For `phi[1:nx_total, ...]` the x-indices are `1, ..., nx_total-1` (20 elements for nx_total=21).
    # For `phi[0:nx_total-1, ...]` the x-indices are `0, ..., nx_total-2` (20 elements for nx_total=21).

    # To strictly match the error message: the issue must be with the y or z slicing.
    # The error (20,20,4) (21,20,4) means phi[1:nx_total, ...] has (20,20,4)
    # and phi[0:nx_total-1, ...] has (21,20,4)
    # This is highly unusual if nx_total=21. This implies one of the slices is actually
    # '0:nx_total' (21 elements) or '1:nx_total+1' (21 elements).

    # Let's ensure the y and z slices for gradient calculation are precisely the interior ones:
    grad_phi_x_at_faces = (
        phi[1:nx_total, 1:ny_total-1, 1:nz_total-1] -  # Explicitly use 1:ny_total-1 and 1:nz_total-1
        phi[0:nx_total-1, 1:ny_total-1, 1:nz_total-1]
    ) / dx
    # Expected shape: (nx_total-1, ny_total-2, nz_total-2) = (20, 19, 3) for 21x21x5

    print(f"[DEBUG pressure_correction.py] grad_phi_x_at_faces pre-check: {grad_phi_x_at_faces.shape}")


    # Gradient of phi in y-direction (for u_y at j+0.5 faces)
    grad_phi_y_at_faces = (
        phi[1:nx_total-1, 1:ny_total, 1:nz_total-1] - # Explicitly use 1:ny_total and 0:ny_total-1 for y
        phi[1:nx_total-1, 0:ny_total-1, 1:nz_total-1]
    ) / dy
    # Expected shape: (nx_total-2, ny_total-1, nz_total-2) = (19, 20, 3)

    print(f"[DEBUG pressure_correction.py] grad_phi_y_at_faces pre-check: {grad_phi_y_at_faces.shape}")


    # Gradient of phi in z-direction (for u_z at k+0.5 faces)
    grad_phi_z_at_faces = (
        phi[1:nx_total-1, 1:ny_total-1, 1:nz_total] - # Explicitly use 1:nz_total and 0:nz_total-1 for z
        phi[1:nx_total-1, 1:ny_total-1, 0:nz_total-1]
    ) / dz
    # Expected shape: (nx_total-2, ny_total-2, nz_total-1) = (19, 19, 4)

    print(f"[DEBUG pressure_correction.py] grad_phi_z_at_faces pre-check: {grad_phi_z_at_faces.shape}")


    # 2. Update velocity field (u_new = u_star - (Δt/ρ)∇φ)
    # Apply correction to X-velocity (u_x)
    # The u-velocity component at index i corresponds to the face between cell i-1 and i.
    # So, the range of interior u-velocities (indices 1 to nx_total-1)
    # matches the range of grad_phi_x_at_faces.
    updated_velocity_field[1:nx_total, 1:ny_total-1, 1:nz_total-1, 0] -= (dt / density) * grad_phi_x_at_faces
    
    # Apply correction to Y-velocity (u_y)
    # Similar logic, v-velocities are from 1 to ny_total-1.
    updated_velocity_field[1:nx_total-1, 1:ny_total, 1:nz_total-1, 1] -= (dt / density) * grad_phi_y_at_faces

    # Apply correction to Z-velocity (u_z)
    # Similar logic, w-velocities are from 1 to nz_total-1.
    updated_velocity_field[1:nx_total-1, 1:ny_total-1, 1:nz_total, 2] -= (dt / density) * grad_phi_z_at_faces

    # 3. Update pressure (P_new = P_old + ρφ)
    # This update is for the interior pressure cells.
    # phi itself is (nx_total, ny_total, nz_total), and we only apply it to the interior.
    updated_pressure[1:nx_total-1, 1:ny_total-1, 1:nz_total-1] += density * phi[1:nx_total-1, 1:ny_total-1, 1:nz_total-1]

    # 4. Re-calculate divergence after correction for debug/validation
    # Compute divergence of the corrected velocity field for interior cells
    divergence_after_correction_field = np.zeros_like(current_pressure_field) # Same shape as pressure

    # For interior cells (1 to N-2)
    # ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
    # ∂u/∂x at cell center (i,j,k) approximated using central difference of u_x at faces: (u_x[i+1/2,j,k] - u_x[i-1/2,j,k]) / dx
    # The u_x values used are updated_velocity_field[i, j, k, 0] and updated_velocity_field[i-1, j, k, 0].
    
    divergence_after_correction_field[1:nx_total-1, 1:ny_total-1, 1:nz_total-1] = (
        (updated_velocity_field[1:nx_total-1, 1:ny_total-1, 1:nz_total-1, 0] - updated_velocity_field[0:nx_total-2, 1:ny_total-1, 1:nz_total-1, 0]) / dx + # u_x
        (updated_velocity_field[1:nx_total-1, 1:ny_total-1, 1:nz_total-1, 1] - updated_velocity_field[1:nx_total-1, 0:ny_total-2, 1:ny_total-1, 1]) / dy + # u_y
        (updated_velocity_field[1:nx_total-1, 1:ny_total-1, 1:nz_total-1, 2] - updated_velocity_field[1:nx_total-1, 1:ny_total-1, 0:nz_total-2, 2]) / dz     # u_z
    )

    return updated_velocity_field, updated_pressure, \
           grad_phi_x_at_faces, grad_phi_y_at_faces, grad_phi_z_at_faces, \
           divergence_after_correction_field



