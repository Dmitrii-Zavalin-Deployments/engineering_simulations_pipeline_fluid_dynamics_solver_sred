import numpy as np
from numba import jit, float64, int32

@jit(float64[:,:,:](float64[:,:,:], float64[:,:,:], float64, float64, float64, float64, float64), nopython=True, parallel=False, cache=True)
def _sor_kernel(phi, b, dx, dy, dz, omega, max_iterations):
    """
    Numba-optimized kernel for the Successive Over-Relaxation (SOR) method.
    This function performs the iterative updates for solving the Poisson equation.
    Operates directly on 3D NumPy arrays.

    Args:
        phi (np.ndarray): The pressure correction field (solution variable), shape (nx, ny, nz).
                          Modified in-place.
        b (np.ndarray): The source term (divergence_u_star / dt), shape (nx, ny, nz).
        dx, dy, dz (float): Grid spacing in each dimension.
        omega (float): Relaxation factor for SOR.
        max_iterations (int): Maximum number of iterations.

    Returns:
        np.ndarray: The updated phi field after max_iterations.
    """
    nx, ny, nz = phi.shape
    # Pre-compute coefficients to avoid repeated division inside the loop
    dx2_inv = 1.0 / (dx**2)
    dy2_inv = 1.0 / (dy**2)
    dz2_inv = 1.0 / (dz**2)
    # Denominator term for SOR update
    denom = 2.0 * (dx2_inv + dy2_inv + dz2_inv)

    for iteration in range(max_iterations):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Handle boundaries
                    if (i == 0 or i == nx - 1 or
                        j == 0 or j == ny - 1 or
                        k == 0 or k == nz - 1):
                        # Dirichlet boundary conditions: assume phi is fixed at 0.
                        # This might need to be configurable later, but for now, it's 0.
                        # The pressure reference point is often set to 0.
                        phi[i, j, k] = 0.0 # Or previous value if Neumann, but we fix at 0
                    else:
                        # Interior points: apply SOR update
                        # Neighbors (excluding self)
                        term_x = (phi[i+1, j, k] + phi[i-1, j, k]) * dx2_inv
                        term_y = (phi[i, j+1, k] + phi[i, j-1, k]) * dy2_inv
                        term_z = (phi[i, j, k+1] + phi[i, j, k-1]) * dz2_inv

                        # RHS: (source term * (1/dt))
                        rhs_term = b[i, j, k]

                        # Standard Jacobi-like update
                        phi_new_val = (term_x + term_y + term_z - rhs_term) / denom

                        # SOR update: phi_new = (1-omega)*phi_old + omega*phi_jacobi
                        phi[i, j, k] = (1.0 - omega) * phi[i, j, k] + omega * phi_new_val
    return phi


def solve_poisson_for_phi(divergence_u_star, mesh_info, time_step,
                          omega=1.7, max_iterations=1000, tolerance=1e-6):
    """
    Solves the Poisson equation for pressure correction (phi) using the SOR method.
    nabla^2(phi) = (1/dt) * (nabla . u*)

    Args:
        divergence_u_star (np.ndarray): The divergence of the tentative velocity field,
                                        shape (nx, ny, nz).
        mesh_info (dict): Dictionary containing grid information:
                          - 'grid_shape': (nx, ny, nz) tuple.
                          - 'dx', 'dy', 'dz': Grid spacing in each dimension.
        time_step (float): The simulation time step (dt).
        omega (float): Relaxation factor for SOR (default 1.7, common for 3D).
        max_iterations (int): Maximum number of iterations for the solver.
        tolerance (float): Convergence tolerance (not yet implemented in Numba kernel).

    Returns:
        np.ndarray: The pressure correction field phi, shape (nx, ny, nz).
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Initialize phi to zeros. This is typically the reference pressure point.
    phi = np.zeros((nx, ny, nz), dtype=np.float64)

    # Source term for Poisson equation: S = (1/dt) * (nabla . u*)
    b_source = divergence_u_star / time_step

    # Call the Numba-optimized SOR kernel
    # The _sor_kernel function modifies phi in-place and returns it.
    phi = _sor_kernel(phi, b_source, dx, dy, dz, omega, max_iterations)

    return phi