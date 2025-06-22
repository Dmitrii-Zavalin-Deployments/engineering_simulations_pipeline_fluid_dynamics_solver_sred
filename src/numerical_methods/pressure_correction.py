import numpy as np

def apply_pressure_correction(u_tentative, current_pressure, phi, mesh_info, time_step, rho):
    """
    Applies the pressure correction to the tentative velocity field and updates the pressure.
    u^(n+1) = u* - dt * grad(phi)
    p^(n+1) = p^n + rho * phi

    Args:
        u_tentative (np.ndarray): The tentative velocity field, shape (nx, ny, nz, 3).
        current_pressure (np.ndarray): The pressure field from the previous time step, shape (nx, ny, nz).
        phi (np.ndarray): The pressure correction field obtained from the Poisson solver, shape (nx, ny, nz).
        mesh_info (dict): Dictionary containing grid information:
                          - 'grid_shape': (nx, ny, nz) tuple.
                          - 'dx', 'dy', 'dz': Grid spacing in each dimension.
        time_step (float): The simulation time step (dt).
        rho (float): The fluid density.

    Returns:
        tuple: (velocity_next (np.ndarray), pressure_next (np.ndarray))
               The corrected velocity field and the updated pressure field.
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Initialize corrected velocity and next pressure
    velocity_next = np.copy(u_tentative) # Start with tentative velocity
    pressure_next = np.copy(current_pressure) # Start with current pressure

    # --- Compute grad(phi) = (dphi/dx, dphi/dy, dphi/dz) ---
    # Using central differences for interior points, and forward/backward for boundaries.

    # dphi/dx
    dphi_dx = np.zeros((nx, ny, nz), dtype=np.float64)
    if nx > 1:
        dphi_dx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
        dphi_dx[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / dx       # Forward difference at x=0
        dphi_dx[nx-1, :, :] = (phi[nx-1, :, :] - phi[nx-2, :, :]) / dx # Backward difference at x=nx-1
    elif nx == 1:
        # If only one point in x, derivative is 0 or needs special BC handling
        pass # dphi_dx remains zeros

    # dphi/dy
    dphi_dy = np.zeros((nx, ny, nz), dtype=np.float64)
    if ny > 1:
        dphi_dy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
        dphi_dy[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / dy       # Forward difference at y=0
        dphi_dy[:, ny-1, :] = (phi[:, ny-1, :] - phi[:, ny-2, :]) / dy # Backward difference at y=ny-1
    elif ny == 1:
        pass # dphi_dy remains zeros

    # dphi/dz
    dphi_dz = np.zeros((nx, ny, nz), dtype=np.float64)
    if nz > 1:
        dphi_dz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dz)
        dphi_dz[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / dz       # Forward difference at z=0
        dphi_dz[:, :, nz-1] = (phi[:, :, nz-1] - phi[:, :, nz-2]) / dz # Backward difference at z=nz-1
    elif nz == 1:
        pass # dphi_dz remains zeros

    # --- Apply velocity correction ---
    # u^(n+1) = u* - dt * dphi/dx
    # v^(n+1) = v* - dt * dphi/dy
    # w^(n+1) = w* - dt * dphi/dz

    # Subtract the pressure gradient term from the respective velocity components
    velocity_next[..., 0] -= time_step * dphi_dx  # X-component (Vx)
    velocity_next[..., 1] -= time_step * dphi_dy  # Y-component (Vy)
    velocity_next[..., 2] -= time_step * dphi_dz  # Z-component (Vz)

    # --- Update pressure ---
    # p^(n+1) = p^n + rho * phi
    pressure_next = current_pressure + rho * phi

    return velocity_next, pressure_next