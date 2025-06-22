import numpy as np

def compute_pressure_divergence(u_tentative, mesh_info):
    """
    Computes the divergence of the tentative velocity field (nabla . u*).
    This function operates on the 3D reshaped NumPy array u_tentative.
    Uses central differences for derivatives.

    Args:
        u_tentative (np.ndarray): The tentative velocity field, shape (nx, ny, nz, 3).
        mesh_info (dict): Dictionary containing grid information:
                          - 'grid_shape': (nx, ny, nz) tuple.
                          - 'dx', 'dy', 'dz': Grid spacing in each dimension.

    Returns:
        np.ndarray: The divergence field, shape (nx, ny, nz).
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Initialize divergence field
    divergence = np.zeros((nx, ny, nz), dtype=np.float64)

    # Extract velocity components
    u = u_tentative[..., 0] # Vx
    v = u_tentative[..., 1] # Vy
    w = u_tentative[..., 2] # Vz

    # --- Compute dU/dX ---
    # Central difference: (u[i+1] - u[i-1]) / (2*dx)
    # Applied to interior points in x-direction.
    if nx > 1:
        # For the interior points (from 1 to nx-2)
        divergence[1:-1, :, :] += (u[2:, :, :] - u[:-2, :, :]) / (2 * dx)
        # Boundary handling for dU/dX at i=0 and i=nx-1
        # Use forward difference at i=0, backward difference at i=nx-1
        # This is a common practice when central difference can't be used at boundaries.
        if nx > 0: # Ensure index exists
            divergence[0, :, :] += (u[1, :, :] - u[0, :, :]) / dx # Forward difference at i=0
        if nx > 1: # Ensure index exists
            divergence[nx-1, :, :] += (u[nx-1, :, :] - u[nx-2, :, :]) / dx # Backward difference at i=nx-1
    elif nx == 1:
        # For a 1D problem in X (only 1 node), dU/dX is implicitly 0 or handled by BCs
        # If it's a point, divergence is 0, so divergence remains zeros.
        pass

    # --- Compute dV/dY ---
    # Central difference: (v[j+1] - v[j-1]) / (2*dy)
    if ny > 1:
        divergence[:, 1:-1, :] += (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)
        if ny > 0:
            divergence[:, 0, :] += (v[:, 1, :] - v[:, 0, :]) / dy # Forward difference at j=0
        if ny > 1:
            divergence[:, ny-1, :] += (v[:, ny-1, :] - v[:, ny-2, :]) / dy # Backward difference at j=ny-1
    elif ny == 1:
        pass

    # --- Compute dW/dZ ---
    # Central difference: (w[k+1] - w[k-1]) / (2*dz)
    if nz > 1:
        divergence[:, :, 1:-1] += (w[:, :, 2:] - w[:, :, :-2]) / (2 * dz)
        if nz > 0:
            divergence[:, :, 0] += (w[:, :, 1] - w[:, :, 0]) / dz # Forward difference at k=0
        if nz > 1:
            divergence[:, :, nz-1] += (w[:, :, nz-1] - w[:, :, nz-2]) / dz # Backward difference at k=nz-1
    elif nz == 1:
        pass

    return divergence