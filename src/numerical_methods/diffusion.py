# src/numerical_methods/diffusion.py

import numpy as np

def compute_diffusion_term(field, viscosity, mesh_info):
    """
    Computes the diffusion term (viscosity * ∇²(field)) using central differences.

    Args:
        field (np.ndarray): Scalar field (nx, ny, nz) or vector field (nx, ny, nz, 3) with ghost cells.
                            The ghost cells are assumed to be correctly populated by boundary conditions.
        viscosity (float): The fluid's kinematic or dynamic viscosity (ν or μ).
        mesh_info (dict): Grid metadata including:
            - 'grid_shape': (nx, ny, nz)
            - 'dx', 'dy', 'dz': spacing in x, y, z directions.

    Returns:
        np.ndarray: Diffusion term (same shape as input field).
    """
    # Extract grid dimensions and spacing
    # Note: grid_shape from mesh_info typically refers to interior cells (nx, ny, nz).
    # The 'field' array itself will have (nx+2, ny+2, nz+2) due to ghost cells.
    # We use the shape of the input 'field' for array operations.
    # nx_field, ny_field, nz_field = field.shape[:3] # This would be (nx+2, ny+2, nz+2)
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Defensive clamping for input field to prevent NaNs/Infs from propagating
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    # Initialize arrays for second derivatives in each direction
    d2x = np.zeros_like(field)
    d2y = np.zeros_like(field)
    d2z = np.zeros_like(field)

    # Compute second derivatives using central differences for the interior cells.
    # These calculations rely on the ghost cells of the 'field' input.

    # Second derivative in x-direction (d^2/dx^2)
    # Applies to cells from index 1 to -2 (inclusive) along the x-axis
    if field.shape[0] > 2: # Ensure there are enough cells for central differencing (i.e., at least 3 cells including ghost)
        d2x[1:-1, :, :] = (field[2:, :, :] - 2 * field[1:-1, :, :] + field[:-2, :, :]) / dx**2
        # Extend the calculated interior derivatives to the boundary cells of d2x
        d2x[0, :, :] = d2x[1, :, :]
        d2x[-1, :, :] = d2x[-2, :, :]

    # Second derivative in y-direction (d^2/dy^2)
    # Applies to cells from index 1 to -2 (inclusive) along the y-axis
    if field.shape[1] > 2:
        d2y[:, 1:-1, :] = (field[:, 2:, :] - 2 * field[:, 1:-1, :] + field[:, :-2, :]) / dy**2
        # Extend the calculated interior derivatives to the boundary cells of d2y
        d2y[:, 0, :] = d2y[:, 1, :]
        d2y[:, -1, :] = d2y[:, -2, :]

    # Second derivative in z-direction (d^2/dz^2)
    # Applies to cells from index 1 to -2 (inclusive) along the z-axis
    if field.shape[2] > 2:
        d2z[:, :, 1:-1] = (field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]) / dz**2
        # Extend the calculated interior derivatives to the boundary cells of d2z
        d2z[:, :, 0] = d2z[:, :, 1]
        d2z[:, :, -1] = d2z[:, :, -2]

    # Total diffusion term is viscosity times the Laplacian (sum of second derivatives)
    diffusion = viscosity * (d2x + d2y + d2z)

    # Final check for NaNs/Infs in the computed diffusion term and clamp if necessary
    if np.isnan(diffusion).any() or np.isinf(diffusion).any():
        print("❌ Warning: Invalid values in diffusion term — clamping to zero.")
    diffusion = np.nan_to_num(diffusion, nan=0.0, posinf=0.0, neginf=0.0)

    return diffusion




