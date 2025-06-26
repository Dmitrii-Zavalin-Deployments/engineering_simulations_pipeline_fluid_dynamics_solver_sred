# src/numerical_methods/pressure_divergence.py
import numpy as np

def compute_pressure_divergence(u_field, mesh_info):
    """
    Computes the divergence of the velocity field (∇·u) using central differencing.
    The velocity field is assumed to include a single layer of ghost cells.

    Args:
        u_field (np.ndarray): The velocity field with ghost cells (nx+2, ny+2, nz+2, 3).
        mesh_info (dict): Dictionary with grid information, including 'grid_shape', 'dx', etc.

    Returns:
        np.ndarray: A 3D array representing the divergence at each cell of the
                    physical grid (nx, ny, nz).
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']
    
    # Extract velocity components from the ghosted field
    # u, v, w are now (nx+2, ny+2, nz+2) arrays
    u, v, w = u_field[..., 0], u_field[..., 1], u_field[..., 2]
    
    # Initialize the divergence array for the physical grid (no ghost cells)
    divergence = np.zeros((nx, ny, nz))
    
    # --- FIX: Corrected slicing for central differencing over the interior grid ---
    # The interior of the ghosted grid is at indices 1 to nx/ny/nz.
    # Central differencing for a cell 'i' uses values at 'i+1' and 'i-1'.
    
    # X-direction derivative (∂u/∂x) using slices [2:-1] and [0:-3] on the ghosted grid
    # These slices correspond to the i+1 and i-1 neighbors for the interior cells (1:-1)
    # in the ghosted grid. The result has shape (nx, ny, nz).
    du_dx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    
    # Y-direction derivative (∂v/∂y)
    dv_dy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    
    # Z-direction derivative (∂w/∂z)
    dw_dz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)
    
    # Sum the derivatives to get the total divergence
    divergence = du_dx + dv_dy + dw_dz
    
    return divergence

def compute_pressure_gradient(p_field, mesh_info):
    """
    Computes the gradient of the scalar pressure field using central differencing.

    Args:
        p_field (np.ndarray): Pressure field with ghost cells (nx+2, ny+2, nz+2).
        mesh_info (dict): Dictionary with grid spacings: 'dx', 'dy', 'dz'.

    Returns:
        np.ndarray: Pressure gradient field of shape (nx, ny, nz, 3).
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    grad = np.zeros(p_field.shape + (3,), dtype=p_field.dtype)

    grad[1:-1, 1:-1, 1:-1, 0] = (p_field[2:, 1:-1, 1:-1] - p_field[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad[1:-1, 1:-1, 1:-1, 1] = (p_field[1:-1, 2:, 1:-1] - p_field[1:-1, :-2, 1:-1]) / (2 * dy)
    grad[1:-1, 1:-1, 1:-1, 2] = (p_field[1:-1, 1:-1, 2:] - p_field[1:-1, 1:-1, :-2]) / (2 * dz)

    return grad



