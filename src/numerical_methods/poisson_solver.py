import numpy as np

def solve_poisson_for_phi(poisson_rhs, mesh_info, num_iterations=100, omega=1.7):
    """
    Solves the Poisson equation (del^2(phi) = S) for the pressure correction field (phi)
    using Successive Over-Relaxation (SOR) method.
    """
    num_nodes = mesh_info["nodes"]
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    idx_to_node = mesh_info["idx_to_node"]
    node_to_idx = mesh_info["node_to_idx"]

    phi = np.zeros(num_nodes) # Initialize phi
    phi_new = np.copy(phi)

    for _iter in range(num_iterations):
        for node_1d_idx in range(num_nodes):
            i, j, k = node_to_idx[node_1d_idx]
            
            phi_sum_neighbors = 0.0
            coeff_diag = 0.0

            # X-direction
            if nx > 1:
                if i > 0: phi_sum_neighbors += phi_new[idx_to_node[(i-1, j, k)]] / (dx**2)
                if i < nx - 1: phi_sum_neighbors += phi[idx_to_node[(i+1, j, k)]] / (dx**2)
                coeff_diag += 2 / (dx**2) if i > 0 and i < nx - 1 else 1 / (dx**2)

            # Y-direction
            if ny > 1:
                if j > 0: phi_sum_neighbors += phi_new[idx_to_node[(i, j-1, k)]] / (dy**2)
                if j < ny - 1: phi_sum_neighbors += phi[idx_to_node[(i, j+1, k)]] / (dy**2)
                coeff_diag += 2 / (dy**2) if j > 0 and j < ny - 1 else 1 / (dy**2)

            # Z-direction
            if nz > 1:
                if k > 0: phi_sum_neighbors += phi_new[idx_to_node[(i, j, k-1)]] / (dz**2)
                if k < nz - 1: phi_sum_neighbors += phi[idx_to_node[(i, j, k+1)]] / (dz**2)
                coeff_diag += 2 / (dz**2) if k > 0 and k < nz - 1 else 1 / (dz**2)

            if coeff_diag > 1e-12:
                phi_new[node_1d_idx] = (1 - omega) * phi[node_1d_idx] + \
                                       omega * ((phi_sum_neighbors - poisson_rhs[node_1d_idx]) / coeff_diag)
            else:
                phi_new[node_1d_idx] = 0.0
            
        phi = np.copy(phi_new)
    return phi