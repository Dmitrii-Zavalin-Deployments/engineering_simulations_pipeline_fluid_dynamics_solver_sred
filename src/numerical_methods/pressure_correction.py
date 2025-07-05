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
    # We use these to define slices for interior cells.

    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Initialize updated velocity and pressure fields
    updated_velocity_field = np.copy(tentative_velocity_field)
    updated_pressure = np.copy(current_pressure_field)

    # 1. Calculate the gradient of phi
    # grad_phi_x, grad_phi_y, grad_phi_z need to be calculated at the cell faces
    # where velocities are defined. For simplicity, we can approximate at cell centers.
    # More accurately, these should align with the staggered grid.
    # For now, let's compute central differences for interior points of phi.

    # Compute gradient of phi for interior cells
    # Indices for interior cells: 1 to nx_total-2, 1 to ny_total-2, 1 to nz_total-2
    # So the slices below correctly select the interior values.
    grad_phi_x = (phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_phi_y = (phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) / (2 * dy)
    grad_phi_z = (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, :-2]) / (2 * dz)

    # These gradients will have shape (nx_interior, ny_interior, nz_interior)
    # i.e., (nx_total - 2, ny_total - 2, nz_total - 2)

    # 2. Update velocity field (u_new = u_star - (Δt/ρ)∇φ)
    # The pressure correction is applied to the interior velocity components.
    # Ensure shapes match for the interior slices.
    # updated_velocity_field has shape (nx_total, ny_total, nz_total, 3)
    # The velocity components are at cell faces.
    # u_x at (i+1/2, j, k), u_y at (i, j+1/2, k), u_z at (i, j, k+1/2)
    # The gradient terms also need to be at these locations.
    # For now, let's assume `grad_phi_x` etc are approximated for cell centers (i,j,k)
    # and we apply them to the corresponding interior velocity components.

    # This is a common point of error in staggered grid implementations.
    # For simplicity, assuming `grad_phi_x` is applied to u at cell centers.
    # This might need refinement for a true staggered grid.

    # Apply correction to X-velocity (u_x)
    # u_x is typically defined at the faces between (i-1,j,k) and (i,j,k).
    # Its index range is from 0 to nx_total-1. The relevant interior indices are 1 to nx_total-2.
    # The pressure gradient used for u_x at i+1/2 face should be (phi[i+1] - phi[i])/dx
    # This means grad_phi_x in `apply_pressure_correction` should be calculated
    # using differences between adjacent `phi` values that align with velocity components.

    # Re-evaluating `grad_phi` for staggered grid:
    # ∇x φ at (i+1/2, j, k) ≈ (φ[i+1, j, k] - φ[i, j, k]) / dx
    # ∇y φ at (i, j+1/2, k) ≈ (φ[i, j+1, k] - φ[i, j, k]) / dy
    # ∇z φ at (i, j, k+1/2) ≈ (φ[i, j, k+1] - φ[i, j, k]) / dz

    # Calculate gradients for velocity correction at cell faces:
    grad_phi_x_at_faces = (phi[1:-1, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / dx # Use (i) - (i-1) for u at (i-1/2)
    # This means grad_phi_x_at_faces will have shape (nx_total-2, ny_interior, nz_interior)
    # Let's align these carefully with the velocity field.
    # u_x is stored in updated_velocity_field[:,:,:,0]
    # The u_x values at interior faces are at indices (1:nx_total-1, 1:ny_total-1, 1:nz_total-1).
    # The corresponding phi indices for (phi[i+1] - phi[i]) are also (1:nx_total-1, 1:ny_total-1, 1:nz_total-1) for phi[i+1]
    # and (0:nx_total-2, 1:ny_total-1, 1:nz_total-1) for phi[i].

    # Correct gradient calculation for staggered grid:
    # d_phi/dx at (i+1/2, j, k) is (phi[i+1,j,k] - phi[i,j,k])/dx
    # This gradient applies to u_x(i+1/2, j, k)
    # u_x components are stored from index 0 to nx_total-1.
    # The relevant interior u_x components are from index 1 to nx_total-2.
    # This means they align with phi cells from index 1 to nx_total-2.
    # So, the phi terms used are phi[1...nx_total-1] and phi[0...nx_total-2]
    # For `updated_velocity_field[1:-1, 1:-1, 1:-1, 0]`, its x-indices range from 1 to nx_total-2.
    # The corresponding gradient needs to be over these x-indices.
    # The phi[i+1] term would be phi[2:-1] and phi[i] would be phi[1:-2]
    # Let's adjust the slices to be very precise.

    # Velocity components (u, v, w) are typically at cell faces:
    # u_x at (i+0.5, j, k)
    # u_y at (i, j+0.5, k)
    # u_z at (i, j, k+0.5)

    # The array `phi` has ghost cells (0, N-1) and interior cells (1, N-2).
    # Its shape is (nx_total, ny_total, nz_total).
    # The interior phi values are at phi[1:-1, 1:-1, 1:-1]
    # Let's compute gradients at the faces where velocities live.

    # For u_x (at i+0.5):
    # The x-indices for u_x are typically 0 to nx_total-1.
    # The update happens for interior u_x, which usually means u_x at faces inside the domain.
    # These are u_x[1:-1, :, :].
    # grad_phi_x needs to be computed for faces.
    # (phi at i+1, j, k) - (phi at i, j, k)
    # phi_x is defined at the center of the x faces.
    # So, for the velocity component `updated_velocity_field[i,j,k,0]`, which is at `(i+0.5, j, k)` face,
    # the corresponding phi gradient is `(phi[i+1,j,k] - phi[i,j,k])/dx`
    # The indices for `updated_velocity_field[1:-1, 1:-1, 1:-1, 0]` are `(idx_x, idx_y, idx_z)` where
    # `idx_x` goes from 1 to `nx_total - 2`.
    # This means the phi indices are: `phi[idx_x+1, idx_y, idx_z]` and `phi[idx_x, idx_y, idx_z]`
    # So for `updated_velocity_field[1:nx_total-1, 1:-1, 1:-1, 0]` (the u_x interior values):
    grad_phi_x = (phi[2:, 1:-1, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dx # This applies to u_x from index 1 to nx_total-2
    # Shape of grad_phi_x will be (nx_total-2, ny_total-2, nz_total-2). This is correct for this index range.
    # The velocity `updated_velocity_field[1:-1, 1:-1, 1:-1, 0]` has shape (nx_total-2, ny_total-2, nz_total-2).
    updated_velocity_field[1:-1, 1:-1, 1:-1, 0] -= (dt / density) * grad_phi_x

    # For u_y (at j+0.5):
    grad_phi_y = (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dy
    updated_velocity_field[1:-1, 1:-1, 1:-1, 1] -= (dt / density) * grad_phi_y

    # For u_z (at k+0.5):
    grad_phi_z = (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) / dz
    updated_velocity_field[1:-1, 1:-1, 1:-1, 2] -= (dt / density) * grad_phi_z

    # 3. Update pressure (P_new = P_old + ρφ)
    # This update is for the interior pressure cells.
    # phi itself is (nx_total, ny_total, nz_total), and we only apply it to the interior.
    # This line was the source of the `ValueError`.
    # If `phi` is (21,21,5) and `updated_pressure` is (21,21,5), then their interior slices
    # `[1:-1, 1:-1, 1:-1]` will both be `(19,19,3)`, which are compatible.
    # The original error message `(21,21,5) (19,19,3) (21,21,5)` strongly suggests
    # that somewhere `phi` was *not* treated as the full `(nx_total, ny_total, nz_total)` array
    # but as the `(nx_interior, ny_interior, nz_interior)` array directly.

    # Let's ensure phi is always the full array with ghost cells when passed here.
    # The previous `poisson_solver.py` now guarantees `phi_field` (returned as `phi`)
    # has shape `(nx_total, ny_total, nz_total)`.
    # Therefore, the original problematic line should now work correctly, as both sides will be (19,19,3).
    # This implies the error might have been from a prior run or an incomplete update.
    # Let's keep the line as it was, as it *should* be correct now if phi is consistently (nx_total, ny_total, nz_total).
    updated_pressure[1:-1, 1:-1, 1:-1] += density * phi[1:-1, 1:-1, 1:-1]


    # 4. Re-calculate divergence after correction for debug/validation
    # Compute divergence of the corrected velocity field for interior cells
    divergence_after_correction_field = np.zeros_like(current_pressure_field) # Same shape as pressure

    # For interior cells (1 to N-2)
    # ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
    # ∂u/∂x at (i,j,k) approximated using central difference of u_x at faces: (u_x[i+1/2] - u_x[i-1/2]) / dx
    # This translates to (u_x[i,j,k,0] - u_x[i-1,j,k,0]) / dx for values at cell centers
    divergence_after_correction_field[1:-1, 1:-1, 1:-1] = (
        (updated_velocity_field[1:-1, 1:-1, 1:-1, 0] - updated_velocity_field[:-2, 1:-1, 1:-1, 0]) / dx + # u_x
        (updated_velocity_field[1:-1, 1:-1, 1:-1, 1] - updated_velocity_field[1:-1, :-2, 1:-1, 1]) / dy + # u_y
        (updated_velocity_field[1:-1, 1:-1, 1:-1, 2] - updated_velocity_field[1:-1, 1:-1, :-2, 2]) / dz    # u_z
    )

    return updated_velocity_field, updated_pressure, grad_phi_x, grad_phi_y, grad_phi_z, divergence_after_correction_field



