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
    nx_total, ny_total, nz_total = current_pressure_field.shape
    # These are the *total* dimensions including ghost cells.

    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Initialize updated velocity and pressure fields
    updated_velocity_field = np.copy(tentative_velocity_field)
    updated_pressure = np.copy(current_pressure_field)

    # Define interior slices for clarity and consistency.
    # Interior dimensions (excluding 1 layer of ghost cells)
    nx_interior = nx_total - 2
    ny_interior = ny_total - 2
    nz_interior = nz_total - 2

    # Slice for cell centers (excluding ghost cells) for pressure update
    cell_center_slice = (slice(1, nx_total - 1), slice(1, ny_total - 1), slice(1, nz_total - 1))

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
    grad_phi_x_at_faces = (
        phi[1:nx_total, cell_center_slice[1], cell_center_slice[2]] -
        phi[0:nx_total-1, cell_center_slice[1], cell_center_slice[2]]
    ) / dx
    # Expected shape: (nx_total-1, ny_interior, nz_interior)
    # For 21x21x5, this is (20, 19, 3)

    # Gradient of phi in y-direction (for u_y at j+0.5 faces)
    # This calculation creates a gradient array that has 'ny_total - 1' elements in y.
    grad_phi_y_at_faces = (
        phi[cell_center_slice[0], 1:ny_total, cell_center_slice[2]] -
        phi[cell_center_slice[0], 0:ny_total-1, cell_center_slice[2]]
    ) / dy
    # Expected shape: (nx_interior, ny_total-1, nz_interior)
    # For 21x21x5, this is (19, 20, 3)

    # Gradient of phi in z-direction (for u_z at k+0.5 faces)
    # This calculation creates a gradient array that has 'nz_total - 1' elements in z.
    grad_phi_z_at_faces = (
        phi[cell_center_slice[0], cell_center_slice[1], 1:nz_total] -
        phi[cell_center_slice[0], cell_center_slice[1], 0:nz_total-1]
    ) / dz
    # Expected shape: (nx_interior, ny_interior, nz_total-1)
    # For 21x21x5, this is (19, 19, 4)

    # 2. Update velocity field (u_new = u_star - (Δt/ρ)∇φ)
    # Apply correction to X-velocity (u_x)
    # The u-velocity component at index i corresponds to the face between cell i-1 and i.
    # So, the range of interior u-velocities (indices 1 to nx_total-1)
    # matches the range of grad_phi_x_at_faces.
    # For nx_total = 21, the slice is 1:21. This implies u[1] to u[20].
    # This corresponds to 20 elements, which matches the (20,19,3) shape of grad_phi_x_at_faces.
    updated_velocity_field[1:nx_total, cell_center_slice[1], cell_center_slice[2], 0] -= (dt / density) * grad_phi_x_at_faces
    
    # Apply correction to Y-velocity (u_y)
    # Similar logic, v-velocities are from 1 to ny_total-1.
    updated_velocity_field[cell_center_slice[0], 1:ny_total, cell_center_slice[2], 1] -= (dt / density) * grad_phi_y_at_faces

    # Apply correction to Z-velocity (u_z)
    # Similar logic, w-velocities are from 1 to nz_total-1.
    updated_velocity_field[cell_center_slice[0], cell_center_slice[1], 1:nz_total, 2] -= (dt / density) * grad_phi_z_at_faces

    # 3. Update pressure (P_new = P_old + ρφ)
    # This update is for the interior pressure cells.
    # phi itself is (nx_total, ny_total, nz_total), and we only apply it to the interior.
    # Both `updated_pressure[1:-1, 1:-1, 1:-1]` and `phi[1:-1, 1:-1, 1:-1]` will now be of shape (nx_interior, ny_interior, nz_interior)
    updated_pressure[cell_center_slice] += density * phi[cell_center_slice]

    # 4. Re-calculate divergence after correction for debug/validation
    # Compute divergence of the corrected velocity field for interior cells
    divergence_after_correction_field = np.zeros_like(current_pressure_field) # Same shape as pressure

    # For interior cells (1 to N-2)
    # ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
    # ∂u/∂x at cell center (i,j,k) approximated using central difference of u_x at faces: (u_x[i+1/2,j,k] - u_x[i-1/2,j,k]) / dx
    # This translates to (updated_velocity_field[i+1,j,k,0] - updated_velocity_field[i,j,k,0]) / dx for divergence at cell centers (i+0.5, j, k)
    # No, divergence is at cell centers (i,j,k). So (u_x at i+1/2 - u_x at i-1/2) / dx.
    # The u_x values used are updated_velocity_field[i, j, k, 0] and updated_velocity_field[i-1, j, k, 0].
    
    divergence_after_correction_field[cell_center_slice] = (
        (updated_velocity_field[1:nx_total-1, cell_center_slice[1], cell_center_slice[2], 0] - updated_velocity_field[0:nx_total-2, cell_center_slice[1], cell_center_slice[2], 0]) / dx + # u_x
        (updated_velocity_field[cell_center_slice[0], 1:ny_total-1, cell_center_slice[2], 1] - updated_velocity_field[cell_center_slice[0], 0:ny_total-2, cell_center_slice[2], 1]) / dy + # u_y
        (updated_velocity_field[cell_center_slice[0], cell_center_slice[1], 1:nz_total-1, 2] - updated_velocity_field[cell_center_slice[0], cell_center_slice[1], 0:nz_total-2, 2]) / dz     # u_z
    )

    # For the return values `grad_phi_x`, `grad_phi_y`, `grad_phi_z`, it's often useful to return them
    # at cell centers for debugging or visualization, or at the faces they were computed for.
    # Let's return them at the faces where they were used for velocity correction.
    return updated_velocity_field, updated_pressure, \
           grad_phi_x_at_faces, grad_phi_y_at_faces, grad_phi_z_at_faces, \
           divergence_after_correction_field



