# src/numerical_methods/diffusion.py

import numpy as np

def _check_nan_inf(field, name, step_number, output_frequency_steps):
    """Helper to check for NaN/Inf and print debug info conditionally."""
    # Only print debug info if the current step is an output step
    if step_number % output_frequency_steps == 0:
        has_nan = np.isnan(field).any()
        has_inf = np.isinf(field).any()
        min_val = np.min(field) if not has_nan and not has_inf else float('nan')
        max_val = np.max(field) if not has_nan and not has_inf else float('nan')
        print(f"  [Diffusion DEBUG] {name} stats BEFORE clamp: min={min_val:.2e}, max={max_val:.2e}, has_nan={has_nan}, has_inf={has_inf}")

    # Always clamp to prevent propagation, regardless of output frequency
    if np.isnan(field).any() or np.isinf(field).any():
        if step_number % output_frequency_steps == 0: # Only print warning if it's an output step
            print(f"  ❌ Warning: Invalid values in {name} — clamping to zero.")
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    if step_number % output_frequency_steps == 0: # Only print debug info if it's an output step
        print(f"  [Diffusion DEBUG] {name} stats AFTER clamp: min={np.min(field):.2e}, max={np.max(field):.2e}")
    return field


def compute_diffusion_term(field, nu, dx, dy, dz, step_number, output_frequency_steps):
    """
    Computes the diffusion term (Laplacian) for a scalar field.
    Assumes the input 'field' already includes ghost cells.
    The diffusion term is computed for the interior cells of the domain.

    Args:
        field (np.ndarray): Scalar field with ghost cells (e.g., u, v, or w component).
                            Shape: (nx+2, ny+2, nz+2).
        nu (float): Kinematic viscosity.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        dz (float): Grid spacing in z-direction.
        step_number (int): Current simulation step number.
        output_frequency_steps (int): Frequency for printing debug output.

    Returns:
        np.ndarray: Diffusion term for the interior cells.
                    Shape: (nx, ny, nz).
    """
    # Check and clamp input field before calculations
    field = _check_nan_inf(field, "diffusion_field input", step_number, output_frequency_steps)

    # Compute second derivatives using central difference for interior cells
    # d^2/dx^2
    laplacian_x = (field[2:, 1:-1, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / (dx**2)
    # d^2/dy^2
    laplacian_y = (field[1:-1, 2:, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / (dy**2)
    # d^2/dz^2
    laplacian_z = (field[1:-1, 1:-1, 2:] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / (dz**2)

    diffusion_term = nu * (laplacian_x + laplacian_y + laplacian_z)

    # Clamp the diffusion term for safety before returning
    if np.isnan(diffusion_term).any() or np.isinf(diffusion_term).any():
        if step_number % output_frequency_steps == 0: # Only print warning if it's an output step
            print(f"  ❌ Warning: Invalid values in diffusion_term — clamping to zero.")
        diffusion_term = np.nan_to_num(diffusion_term, nan=0.0, posinf=0.0, neginf=0.0)

    if step_number % output_frequency_steps == 0: # Only print debug info if it's an output step
        print(f"  [Diffusion DEBUG] diffusion_term output stats: min={np.min(diffusion_term):.2e}, max={np.max(diffusion_term):.2e}")

    return diffusion_term



