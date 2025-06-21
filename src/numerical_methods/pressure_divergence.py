import numpy as np

def compute_velocity_divergence(u_tentative, mesh_info):
    """
    Calculates the divergence of the tentative velocity field.
    This serves as the source term for the Poisson equation.
    """
    num_nodes = mesh_info["nodes"]
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    idx_to_node = mesh_info["idx_to_node"]
    node_to_idx = mesh_info["node_to_idx"]

    divergence_ut = np.zeros(num_nodes)

    for node_1d_idx in range(num_nodes):
        i, j, k = node_to_idx[node_1d_idx]
        
        div_x = 0.0
        if nx > 1:
            if i > 0 and i < nx - 1:
                div_x = (u_tentative[idx_to_node[(i+1, j, k)], 0] - u_tentative[idx_to_node[(i-1, j, k)], 0]) / (2 * dx)
            elif i == 0:
                div_x = (u_tentative[idx_to_node[(i+1, j, k)], 0] - u_tentative[node_1d_idx, 0]) / dx
            elif i == nx - 1:
                div_x = (u_tentative[node_1d_idx, 0] - u_tentative[idx_to_node[(i-1, j, k)], 0]) / dx

        div_y = 0.0
        if ny > 1:
            if j > 0 and j < ny - 1:
                div_y = (u_tentative[idx_to_node[(i, j+1, k)], 1] - u_tentative[idx_to_node[(i, j-1, k)], 1]) / (2 * dy)
            elif j == 0:
                div_y = (u_tentative[idx_to_node[(i, j+1, k)], 1] - u_tentative[node_1d_idx, 1]) / dy
            elif j == ny - 1:
                div_y = (u_tentative[node_1d_idx, 1] - u_tentative[idx_to_node[(i, j-1, k)], 1]) / dy

        div_z = 0.0
        if nz > 1:
            if k > 0 and k < nz - 1:
                div_z = (u_tentative[idx_to_node[(i, j, k+1)], 2] - u_tentative[idx_to_node[(i, j, k-1)], 2]) / (2 * dz)
            elif k == 0:
                div_z = (u_tentative[idx_to_node[(i, j, k+1)], 2] - u_tentative[node_1d_idx, 2]) / dz
            elif k == nz - 1:
                div_z = (u_tentative[node_1d_idx, 2] - u_tentative[idx_to_node[(i, j, k-1)], 2]) / dz

        divergence_ut[node_1d_idx] = div_x + div_y + div_z
    
    return divergence_ut