import numpy as np

def compute_advection_term(u_field, velocity_field, mesh_info):
    """
    Computes the advection term - (u . grad)u using a first-order upwind scheme.
    This function operates on 3D reshaped NumPy arrays for efficiency.

    Args:
        u_field (np.ndarray): The scalar or vector field being advected (e.g., component of velocity).
                              Expected shape (nx, ny, nz) for scalar, or (nx, ny, nz, 3) for vector.
        velocity_field (np.ndarray): The velocity field (Ux, Uy, Uz) at each grid point.
                                     Expected shape (nx, ny, nz, 3).
        mesh_info (dict): Dictionary containing grid information:
                          - 'grid_shape': (nx, ny, nz) tuple.
                          - 'dx', 'dy', 'dz': Grid spacing in each dimension.

    Returns:
        np.ndarray: The computed advection term, same shape as u_field.
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    advection_term = np.zeros_like(u_field)

    # Determine if u_field is a scalar (e.g., pressure or a single velocity component)
    # or a vector (e.g., the full velocity vector for the advection of momentum)
    is_scalar_field = (u_field.ndim == 3)

    # Extract velocity components for easier access
    # These are (nx, ny, nz) arrays
    vel_x = velocity_field[..., 0]
    vel_y = velocity_field[..., 1]
    vel_z = velocity_field[..., 2]

    # --- Compute advection for X-component ---
    # d(u*u)/dx, d(u*v)/dy, d(u*w)/dz for the u (x-component of velocity) equation
    # or similar terms if u_field is a scalar

    # Iterate through each dimension (x, y, z) to compute contributions
    # For each dimension, apply upwind differencing based on the sign of the velocity component
    # This involves creating shifted versions of the field and applying masks.

    # Advection in X-direction
    # d(u_field * vel_x) / dx
    # Upwind for u_field * vel_x term
    # Flow from left (i-1 to i) if vel_x > 0
    # Flow from right (i+1 to i) if vel_x < 0

    # Terms for d(F_x)/dx where F_x = u_field * vel_x
    F_x_at_faces_plus_half = np.zeros_like(u_field) # F_x(i+1/2, j, k)
    F_x_at_faces_minus_half = np.zeros_like(u_field) # F_x(i-1/2, j, k)

    # Positive velocity: use value from left cell (i)
    # Need to handle dimensions for scalar vs vector u_field
    if is_scalar_field:
        F_x_pos = u_field * np.maximum(0, vel_x)
        F_x_neg = u_field * np.minimum(0, vel_x)
    else: # u_field is a vector (e.g., for momentum advection)
        F_x_pos = u_field * np.maximum(0, vel_x[:, :, :, np.newaxis])
        F_x_neg = u_field * np.minimum(0, vel_x[:, :, :, np.newaxis])


    # For F_x(i+1/2, j, k): Use F_x_pos from current cell (i), F_x_neg from next cell (i+1)
    # F_x(i+1/2) = u_i * max(0, vel_x_i+1/2) + u_i+1 * min(0, vel_x_i+1/2)
    # For simplicity, assuming cell-centered velocities represent face velocities
    # This is a common simplification for first-order upwind.
    F_x_at_faces_plus_half[:-1, :, :] = F_x_pos[:-1, :, :] + F_x_neg[1:, :, :]
    # Boundary at i=nx-1 (rightmost face) handled by not extending F_x_neg beyond
    # The term F_x_at_faces_plus_half for the last cell means F_x(nx-1+1/2, j, k)
    # which is the outflow face of the last cell, used to calculate its advection term.
    # For the last slice, we just use the current cell's positive flux for outflow
    # or previous cell's negative flux for inflow. This needs careful consideration for boundaries.
    # For simplicity and robust first-order, we can assume F_x_at_faces_plus_half[-1,:,:]
    # uses F_x_pos[-1,:,:]
    F_x_at_faces_plus_half[-1, :, :] = F_x_pos[-1, :, :] # This is a simple outflow assumption


    # For F_x(i-1/2, j, k): Use F_x_pos from previous cell (i-1), F_x_neg from current cell (i)
    # F_x(i-1/2) = u_i-1 * max(0, vel_x_i-1/2) + u_i * min(0, vel_x_i-1/2)
    F_x_at_faces_minus_half[1:, :, :] = F_x_pos[1:, :, :] + F_x_neg[:-1, :, :]
    # Boundary at i=0 (leftmost face) handled by not extending F_x_pos beyond
    F_x_at_faces_minus_half[0, :, :] = F_x_neg[0, :, :] # This is a simple inflow assumption


    # Advection X-component contribution: (F_x(i+1/2) - F_x(i-1/2)) / dx
    advection_term += (F_x_at_faces_plus_half - F_x_at_faces_minus_half) / dx


    # Advection in Y-direction
    # d(u_field * vel_y) / dy
    F_y_at_faces_plus_half = np.zeros_like(u_field)
    F_y_at_faces_minus_half = np.zeros_like(u_field)

    if is_scalar_field:
        F_y_pos = u_field * np.maximum(0, vel_y)
        F_y_neg = u_field * np.minimum(0, vel_y)
    else:
        F_y_pos = u_field * np.maximum(0, vel_y[:, :, :, np.newaxis])
        F_y_neg = u_field * np.minimum(0, vel_y[:, :, :, np.newaxis])


    F_y_at_faces_plus_half[:, :-1, :] = F_y_pos[:, :-1, :] + F_y_neg[:, 1:, :]
    F_y_at_faces_plus_half[:, -1, :] = F_y_pos[:, -1, :] # Outflow assumption

    F_y_at_faces_minus_half[:, 1:, :] = F_y_pos[:, 1:, :] + F_y_neg[:, :-1, :]
    F_y_at_faces_minus_half[:, 0, :] = F_y_neg[:, 0, :] # Inflow assumption

    advection_term += (F_y_at_faces_plus_half - F_y_at_faces_minus_half) / dy

    # Advection in Z-direction
    # d(u_field * vel_z) / dz
    F_z_at_faces_plus_half = np.zeros_like(u_field)
    F_z_at_faces_minus_half = np.zeros_like(u_field)

    if is_scalar_field:
        F_z_pos = u_field * np.maximum(0, vel_z)
        F_z_neg = u_field * np.minimum(0, vel_z)
    else:
        F_z_pos = u_field * np.maximum(0, vel_z[:, :, :, np.newaxis])
        F_z_neg = u_field * np.minimum(0, vel_z[:, :, :, np.newaxis])

    F_z_at_faces_plus_half[:, :, :-1] = F_z_pos[:, :, :-1] + F_z_neg[:, :, 1:]
    F_z_at_faces_plus_half[:, :, -1] = F_z_pos[:, :, -1] # Outflow assumption

    F_z_at_faces_minus_half[:, :, 1:] = F_z_pos[:, :, 1:] + F_z_neg[:, :, :-1]
    F_z_at_faces_minus_half[:, :, 0] = F_z_neg[:, :, 0] # Inflow assumption

    advection_term += (F_z_at_faces_plus_half - F_z_at_faces_minus_half) / dz

    return advection_term