import numpy as np

def correct_velocity_and_pressure(u_tentative, pressure, phi, mesh_info, dt, density):
    """
    Corrects the tentative velocity field and updates the pressure using the pressure correction (phi).
    """
    num_nodes = mesh_info["nodes"]
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    idx_to_node = mesh_info["idx_to_node"]
    node_to_idx = mesh_info["node_to_idx"]

    for node_1d_idx in range(num_nodes):
        i, j, k = node_to_idx[node_1d_idx]

        dphi_dx, dphi_dy, dphi_dz = 0.0, 0.0, 0.0

        # Gradient of phi (dphi/dx)
        if nx > 1:
            if i > 0 and i < nx - 1:
                dphi_dx = (phi[idx_to_node[(i+1, j, k)]] - phi[idx_to_node[(i-1, j, k)]]) / (2 * dx)
            elif i == 0:
                dphi_dx = (phi[idx_to_node[(i+1, j, k)]] - phi[node_1d_idx]) / dx
            elif i == nx - 1:
                dphi_dx = (phi[node_1d_idx] - phi[idx_to_node[(i-1, j, k)]]) / dx

        # Gradient of phi (dphi/dy)
        if ny > 1:
            if j > 0 and j < ny - 1:
                dphi_dy = (phi[idx_to_node[(i, j+1, k)]] - phi[idx_to_node[(i, j-1, k)]]) / (2 * dy)
            elif j == 0:
                dphi_dy = (phi[idx_to_node[(i, j+1, k)]] - phi[node_1d_idx]) / dy
            elif j == ny - 1:
                dphi_dy = (phi[node_1d_idx] - phi[idx_to_node[(i, j-1, k)]]) / dy

        # Gradient of phi (dphi/dz)
        if nz > 1:
            if k > 0 and k < nz - 1:
                dphi_dz = (phi[idx_to_node[(i, j, k+1)]] - phi[idx_to_node[(i, j, k-1)]]) / (2 * dz)
            elif k == 0:
                dphi_dz = (phi[idx_to_node[(i, j, k+1)]] - phi[node_1d_idx]) / dz
            elif k == nz - 1:
                dphi_dz = (phi[node_1d_idx] - phi[idx_to_node[(i, j, k-1)]]) / dz

        u_tentative[node_1d_idx, 0] -= dt * dphi_dx / density
        u_tentative[node_1d_idx, 1] -= dt * dphi_dy / density
        u_tentative[node_1d_idx, 2] -= dt * dphi_dz / density
    
    new_pressure = pressure + phi * (density / dt)

    return u_tentative, new_pressure