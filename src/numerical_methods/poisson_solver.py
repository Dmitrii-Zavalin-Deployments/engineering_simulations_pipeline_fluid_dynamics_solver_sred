# src/numerical_methods/poisson_solver.py

import numpy as np

def solve_poisson_for_phi(
    divergence_field: np.ndarray,
    mesh_info: dict,
    dt: float,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    return_residual: bool = False,
    should_log_verbose: bool = False # Added verbose logging flag
) -> np.ndarray:
    """
    Solves the Poisson equation ∇²phi = (1/dt) * (∇·u*) using Jacobi iteration.
    The solution 'phi' is a scalar potential used for pressure correction.

    Args:
        divergence_field (np.ndarray): The divergence of the tentative velocity field (∇·u*).
                                       Shape (nx, ny, nz), representing interior cells.
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz', and 'nx', 'ny', 'nz'.
        dt (float): Time step size.
        tolerance (float): Convergence tolerance for the Jacobi iteration.
        max_iterations (int): Maximum number of iterations for the Jacobi solver.
        return_residual (bool): If True, also return the final L2 norm of the residual.
        should_log_verbose (bool): If True, print detailed iteration logs.

    Returns:
        np.ndarray: The scalar potential 'phi' field of shape (nx+2, ny+2, nz+2),
                    including zero ghost cells.
        float (optional): The final L2 norm of the residual if return_residual is True.
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']
    nx, ny, nz = mesh_info['nx'], mesh_info['ny'], mesh_info['nz']

    # Initialize phi with zeros (including ghost cells)
    phi = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64)

    # The source term for the Poisson equation: RHS = (1/dt) * (∇·u*)
    # Make sure to scale the divergence correctly, and ensure it's clamped
    source_term = divergence_field / dt
    source_term = np.nan_to_num(source_term, nan=0.0, posinf=0.0, neginf=0.0)

    # Pre-calculate constants for Jacobi iteration denominator
    # This assumes constant grid spacing for simplicity.
    # The denominator for a central difference Laplacian (∂²/∂x² + ∂²/∂y² + ∂²/∂z²)
    # is -2/dx^2 - 2/dy^2 - 2/dz^2
    # So, the inverse is 1 / (-2 * (1/dx^2 + 1/dy^2 + 1/dz^2))
    denom_inv = 1.0 / (-2.0 * (1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)))

    if should_log_verbose:
        print(f"    - Poisson Solver: Starting Jacobi iteration (Max Iter: {max_iterations}, Tol: {tolerance:.2e})")

    residual = np.inf
    for k in range(max_iterations):
        phi_old = np.copy(phi)

        # Apply Jacobi iteration for interior cells (1 to nx, 1 to ny, 1 to nz)
        # Note: The indices for phi correspond to the full grid (including ghost cells),
        # while source_term corresponds to interior cells (0 to nx-1, etc.).
        # So, (i,j,k) in source_term corresponds to (i+1, j+1, k+1) in phi.
        phi[1:-1, 1:-1, 1:-1] = denom_inv * (
            -source_term
            + (phi_old[2:, 1:-1, 1:-1] + phi_old[:-2, 1:-1, 1:-1]) / (dx*dx)
            + (phi_old[1:-1, 2:, 1:-1] + phi_old[1:-1, :-2, 1:-1]) / (dy*dy)
            + (phi_old[1:-1, 1:-1, 2:] + phi_old[1:-1, 1:-1, :-2]) / (dz*dz)
        )

        # Enforce boundary conditions on phi (Dirichlet, usually phi=0 on boundaries for pressure correction)
        # For simplicity, phi is usually set to zero at boundaries for pressure Poisson equation.
        # This implies that the pressure gradient correction will be applied up to the boundary.
        # Setting ghost cells of phi to be consistent with phi=0 at boundaries (if that's the choice)
        # For now, simply assuming phi values inside the domain are calculated, and ghost cells are 0.
        # If specific Neumann conditions for phi are needed, this would be the place to apply them.
        # For pressure correction, Dirichlet phi=0 is common.
        phi[0, :, :] = 0.0
        phi[-1, :, :] = 0.0
        phi[:, 0, :] = 0.0
        phi[:, -1, :] = 0.0
        phi[:, :, 0] = 0.0
        phi[:, :, -1] = 0.0


        # Calculate residual using the original Poisson equation form
        # R = ∇²phi - (1/dt) * (∇·u*)
        # Interior Laplacian of the current phi
        laplacian_phi = (
            (phi[2:, 1:-1, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]) / (dx*dx) +
            (phi[1:-1, 2:, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, :-2, 1:-1]) / (dy*dy) +
            (phi[1:-1, 1:-1, 2:] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, 1:-1, :-2]) / (dz*dz)
        )
        residual_field = laplacian_phi - source_term

        # L2 norm of the residual
        residual = np.linalg.norm(residual_field) / np.sqrt(nx * ny * nz) # Normalized L2 norm

        if should_log_verbose and (k % 50 == 0 or k == max_iterations - 1 or residual < tolerance):
            print(f"        Iteration {k+1}: Residual = {residual:.6e}")

        if residual < tolerance:
            if should_log_verbose:
                print(f"    - Poisson Solver converged in {k+1} iterations. Final residual: {residual:.6e}")
            break
    else: # This block executes if the loop completes without 'break' (i.e., max_iterations reached)
        print(f"❌ Warning: Poisson Solver did NOT converge after {max_iterations} iterations. Final residual: {residual:.6e}")

    # Ensure phi values are not NaN/Inf after solving
    if np.isnan(phi).any() or np.isinf(phi).any():
        print("❌ Warning: Invalid values in phi after Poisson solve. Clamping to zero.")
        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    if return_residual:
        return phi, residual
    else:
        return phi



