import numpy as np

def compute_diffusion_term(field, viscosity, mesh_info):
    """
    Computes the diffusion term (viscosity * nabla^2(field)) for a given field.
    This function operates on 3D reshaped NumPy arrays for efficiency.
    It uses central differences for the second derivatives.

    Args:
        field (np.ndarray): The scalar or vector field (e.g., velocity component or full velocity).
                            Expected shape (nx, ny, nz) for scalar, or (nx, ny, nz, 3) for vector.
        viscosity (float): The fluid's dynamic viscosity (mu).
        mesh_info (dict): Dictionary containing grid information:
                          - 'grid_shape': (nx, ny, nz) tuple.
                          - 'dx', 'dy', 'dz': Grid spacing in each dimension.

    Returns:
        np.ndarray: The computed diffusion term, same shape as 'field'.
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    diffusion_term = np.zeros_like(field)

    # Determine if field is a scalar (e.g., for pressure, not typically diffused like this)
    # or a vector (e.g., for momentum diffusion)
    is_scalar_field = (field.ndim == 3)

    # --- Compute second derivatives for each dimension ---

    # d^2(field) / dx^2
    # Applies to all (j, k) planes.
    # Central difference: (field[i+1] - 2*field[i] + field[i-1]) / dx^2
    d2_field_dx2 = np.zeros_like(field)
    if nx > 2: # Need at least 3 points for central difference
        d2_field_dx2[1:-1, :, :] = (field[2:, :, :] - 2 * field[1:-1, :, :] + field[:-2, :, :]) / (dx**2)
    elif nx == 2: # Special handling for 2 nodes, effectively a 1st order approx or needs different BC approach
        # This case is tricky for central diff. For now, a simplified approach
        # Or, consider handling 1D/2D scenarios as specific cases earlier.
        # For a 2-node grid, the "interior" doesn't exist for 3-point central diff.
        # For simplicity, if a dimension has only 2 nodes, diffusion might be zero or treated differently.
        # Here, setting to zero for now for d2_dx2 where nx=2, as proper 3-point stencil is not applicable.
        pass # d2_field_dx2 remains zeros

    # d^2(field) / dy^2
    d2_field_dy2 = np.zeros_like(field)
    if ny > 2:
        d2_field_dy2[:, 1:-1, :] = (field[:, 2:, :] - 2 * field[:, 1:-1, :] + field[:, :-2, :]) / (dy**2)
    elif ny == 2:
        pass # d2_field_dy2 remains zeros

    # d^2(field) / dz^2
    d2_field_dz2 = np.zeros_like(field)
    if nz > 2:
        d2_field_dz2[:, :, 1:-1] = (field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]) / (dz**2)
    elif nz == 2:
        pass # d2_field_dz2 remains zeros

    # Sum the components of the Laplacian
    # For a scalar field, nabla^2(scalar) = d2/dx2 + d2/dy2 + d2/dz2
    # For a vector field, nabla^2(vector) = (nabla^2(Vx), nabla^2(Vy), nabla^2(Vz))
    # NumPy handles this broadcasting automatically if the last dimension matches or is absent.
    diffusion_term = viscosity * (d2_field_dx2 + d2_field_dy2 + d2_field_dz2)

    return diffusion_term