# src/numerical_methods/diffusion.py

import numpy as np

def compute_diffusion_term(
    scalar_field: np.ndarray,
    viscosity: float,
    mesh_info: dict,
    should_log_verbose: bool = False # Added verbose logging flag
) -> np.ndarray:
    """
    Computes the diffusion term (ν∇²φ) for a scalar field φ (which can be a velocity component)
    using central differencing.

    Args:
        scalar_field (np.ndarray): The scalar field (e.g., u, v, or w component of velocity)
                                   on the full grid including ghost cells (nx+2, ny+2, nz+2).
        viscosity (float): Kinematic viscosity of the fluid (ν).
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz'.
        should_log_verbose (bool): If True, print detailed debug logs.

    Returns:
        np.ndarray: The computed diffusion term (ν∇²φ) for the interior cells
                    (shape: nx, ny, nz).
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Initialize Laplacian term for interior cells
    laplacian = np.zeros_like(scalar_field[1:-1, 1:-1, 1:-1])

    # Compute second derivatives using central differencing for interior cells
    # ∂²φ/∂x²
    d2phi_dx2 = (scalar_field[2:, 1:-1, 1:-1] - 2 * scalar_field[1:-1, 1:-1, 1:-1] +
                 scalar_field[:-2, 1:-1, 1:-1]) / (dx**2)

    # ∂²φ/∂y²
    d2phi_dy2 = (scalar_field[1:-1, 2:, 1:-1] - 2 * scalar_field[1:-1, 1:-1, 1:-1] +
                 scalar_field[1:-1, :-2, 1:-1]) / (dy**2)

    # ∂²φ/∂z²
    d2phi_dz2 = (scalar_field[1:-1, 1:-1, 2:] - 2 * scalar_field[1:-1, 1:-1, 1:-1] +
                 scalar_field[1:-1, 1:-1, :-2]) / (dz**2)

    # Sum the second derivatives to get the Laplacian (∇²φ)
    laplacian = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

    # Compute the diffusion term (ν∇²φ)
    diffusion_term = viscosity * laplacian

    # Check for NaNs or Infs and handle them
    if np.isnan(diffusion_term).any() or np.isinf(diffusion_term).any():
        print("❌ Warning: NaN or Inf encountered in diffusion term. Clamping to zero.")
        diffusion_term = np.nan_to_num(diffusion_term, nan=0.0, posinf=0.0, neginf=0.0)

    if should_log_verbose:
        # These are debug prints, only show if verbose logging is on
        print(f"    - Diffusion Term Max Abs: {np.max(np.abs(diffusion_term)):.4e}")
        print(f"    - Diffusion Term Mean Abs: {np.mean(np.abs(diffusion_term)):.4e}")

    return diffusion_term



