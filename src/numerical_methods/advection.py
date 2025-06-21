import numpy as np

def compute_advection_acceleration(velocity, mesh_info):
    """
    Computes the advection term (- (u . grad)u) for the Navier-Stokes equations.
    Uses first-order upwind scheme.
    """
    num_nodes = mesh_info["nodes"]
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    idx_to_node = mesh_info["idx_to_node"]
    node_to_idx = mesh_info["node_to_idx"]

    advection_accel = np.zeros_like(velocity)
    
    for node_1d_idx in range(num_nodes):
        i, j, k = node_to_idx[node_1d_idx]
        u_curr, v_curr, w_curr = velocity[node_1d_idx]

        du_dx, du_dy, du_dz = 0.0, 0.0, 0.0
        dv_dx, dv_dy, dv_dz = 0.0, 0.0, 0.0
        dw_dx, dw_dy, dw_dz = 0.0, 0.0, 0.0

        if nx > 1:
            if u_curr >= 0: # Backward difference
                if i > 0:
                    du_dx = (u_curr - velocity[idx_to_node[(i-1, j, k)], 0]) / dx
                    dv_dx = (v_curr - velocity[idx_to_node[(i-1, j, k)], 1]) / dx
                    dw_dx = (w_curr - velocity[idx_to_node[(i-1, j, k)], 2]) / dx
            else: # Forward difference
                if i < nx - 1:
                    du_dx = (velocity[idx_to_node[(i+1, j, k)], 0] - u_curr) / dx
                    dv_dx = (velocity[idx_to_node[(i+1, j, k)], 1] - v_curr) / dx
                    dw_dx = (velocity[idx_to_node[(i+1, j, k)], 2] - w_curr) / dx

        if ny > 1:
            if v_curr >= 0: # Backward difference
                if j > 0:
                    du_dy = (u_curr - velocity[idx_to_node[(i, j-1, k)], 0]) / dy
                    dv_dy = (v_curr - velocity[idx_to_node[(i, j-1, k)], 1]) / dy
                    dw_dy = (w_curr - velocity[idx_to_node[(i, j-1, k)], 2]) / dy
            else: # Forward difference
                if j < ny - 1:
                    du_dy = (velocity[idx_to_node[(i, j+1, k)], 0] - u_curr) / dy
                    dv_dy = (velocity[idx_to_node[(i, j+1, k)], 1] - v_curr) / dy
                    dw_dy = (velocity[idx_to_node[(i, j+1, k)], 2] - w_curr) / dy

        if nz > 1:
            if w_curr >= 0: # Backward difference
                if k > 0:
                    du_dz = (u_curr - velocity[idx_to_node[(i, j, k-1)], 0]) / dz
                    dv_dz = (v_curr - velocity[idx_to_node[(i, j, k-1)], 1]) / dz
                    dw_dz = (w_curr - velocity[idx_to_node[(i, j, k-1)], 2]) / dz
            else: # Forward difference
                if k < nz - 1:
                    du_dz = (velocity[idx_to_node[(i, j, k+1)], 0] - u_curr) / dz
                    dv_dz = (velocity[idx_to_node[(i, j, k+1)], 1] - v_curr) / dz
                    dw_dz = (velocity[idx_to_node[(i, j, k+1)], 2] - w_curr) / dz

        advection_accel[node_1d_idx, 0] = u_curr * du_dx + v_curr * du_dy + w_curr * du_dz
        advection_accel[node_1d_idx, 1] = u_curr * dv_dx + v_curr * dv_dy + w_curr * dv_dz
        advection_accel[node_1d_idx, 2] = u_curr * dw_dx + v_curr * dw_dy + w_curr * dw_dz
    
    return advection_accel