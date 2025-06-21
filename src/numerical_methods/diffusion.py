import numpy as np

def compute_diffusion_acceleration(velocity, mesh_info, viscosity):
    """
    Computes the viscous diffusion term (mu * Laplacian(u)) for the Navier-Stokes equations.
    Handles 1D, 2D, and 3D cases and boundary conditions for second derivatives.
    """
    num_nodes = mesh_info["nodes"]
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    idx_to_node = mesh_info["idx_to_node"]
    node_to_idx = mesh_info["node_to_idx"]

    diffusion_accel = np.zeros_like(velocity)

    def get_laplacian_term(val_center, val_plus1, val_minus1, val_plus2_node_idx, val_minus2_node_idx, h_sq, dim_size, index, velocity_component_idx, velocity_field, idx_to_node_map):
        if dim_size <= 1: # No spatial extent in this dimension
            return 0.0
        if index > 0 and index < dim_size - 1: # Central difference
            return (val_plus1 - 2 * val_center + val_minus1) / h_sq
        elif index == 0: # Forward difference (uses val_center, val_plus1, val_plus2)
            # Fetch val_plus2 using the provided index or fallback to val_plus1 if dim_size is small
            val_plus2 = velocity_field[idx_to_node_map[val_plus2_node_idx], velocity_component_idx] if dim_size > 2 else val_plus1
            return (val_plus2 - 2 * val_plus1 + val_center) / h_sq
        elif index == dim_size - 1: # Backward difference (uses val_center, val_minus1, val_minus2)
            # Fetch val_minus2 using the provided index or fallback to val_minus1 if dim_size is small
            val_minus2 = velocity_field[idx_to_node_map[val_minus2_node_idx], velocity_component_idx] if dim_size > 2 else val_minus1
            return (val_center - 2 * val_minus1 + val_minus2) / h_sq
        return 0.0 # Should not be reached

    for node_1d_idx in range(num_nodes):
        i, j, k = node_to_idx[node_1d_idx]
        u_curr, v_curr, w_curr = velocity[node_1d_idx]

        lap_u, lap_v, lap_w = 0.0, 0.0, 0.0

        # X-direction Laplacian
        if nx > 1:
            u_i_minus_1_val = velocity[idx_to_node[(i-1,j,k)], 0] if i > 0 else u_curr
            u_i_plus_1_val = velocity[idx_to_node[(i+1,j,k)], 0] if i < nx - 1 else u_curr
            v_i_minus_1_val = velocity[idx_to_node[(i-1,j,k)], 1] if i > 0 else v_curr
            v_i_plus_1_val = velocity[idx_to_node[(i+1,j,k)], 1] if i < nx - 1 else v_curr
            w_i_minus_1_val = velocity[idx_to_node[(i-1,j,k)], 2] if i > 0 else w_curr
            w_i_plus_1_val = velocity[idx_to_node[(i+1,j,k)], 2] if i < nx - 1 else w_curr

            lap_u += get_laplacian_term(u_curr, u_i_plus_1_val, u_i_minus_1_val, (min(i+2, nx-1),j,k), (max(i-2, 0),j,k), dx**2, nx, i, 0, velocity, idx_to_node)
            lap_v += get_laplacian_term(v_curr, v_i_plus_1_val, v_i_minus_1_val, (min(i+2, nx-1),j,k), (max(i-2, 0),j,k), dx**2, nx, i, 1, velocity, idx_to_node)
            lap_w += get_laplacian_term(w_curr, w_i_plus_1_val, w_i_minus_1_val, (min(i+2, nx-1),j,k), (max(i-2, 0),j,k), dx**2, nx, i, 2, velocity, idx_to_node)

        # Y-direction Laplacian
        if ny > 1:
            u_j_minus_1_val = velocity[idx_to_node[(i,j-1,k)], 0] if j > 0 else u_curr
            u_j_plus_1_val = velocity[idx_to_node[(i,j+1,k)], 0] if j < ny - 1 else u_curr
            v_j_minus_1_val = velocity[idx_to_node[(i,j-1,k)], 1] if j > 0 else v_curr
            v_j_plus_1_val = velocity[idx_to_node[(i,j+1,k)], 1] if j < ny - 1 else v_curr
            w_j_minus_1_val = velocity[idx_to_node[(i,j-1,k)], 2] if j > 0 else w_curr
            w_j_plus_1_val = velocity[idx_to_node[(i,j+1,k)], 2] if j < ny - 1 else w_curr

            lap_u += get_laplacian_term(u_curr, u_j_plus_1_val, u_j_minus_1_val, (i, min(j+2, ny-1),k), (i, max(j-2, 0),k), dy**2, ny, j, 0, velocity, idx_to_node)
            lap_v += get_laplacian_term(v_curr, v_j_plus_1_val, v_j_minus_1_val, (i, min(j+2, ny-1),k), (i, max(j-2, 0),k), dy**2, ny, j, 1, velocity, idx_to_node)
            lap_w += get_laplacian_term(w_curr, w_j_plus_1_val, w_j_minus_1_val, (i, min(j+2, ny-1),k), (i, max(j-2, 0),k), dy**2, ny, j, 2, velocity, idx_to_node)

        # Z-direction Laplacian
        if nz > 1:
            u_k_minus_1_val = velocity[idx_to_node[(i,j,k-1)], 0] if k > 0 else u_curr
            u_k_plus_1_val = velocity[idx_to_node[(i,j,k+1)], 0] if k < nz - 1 else u_curr
            v_k_minus_1_val = velocity[idx_to_node[(i,j,k-1)], 1] if k > 0 else v_curr
            v_k_plus_1_val = velocity[idx_to_node[(i,j,k+1)], 1] if k < nz - 1 else v_curr
            w_k_minus_1_val = velocity[idx_to_node[(i,j,k-1)], 2] if k > 0 else w_curr
            w_k_plus_1_val = velocity[idx_to_node[(i,j,k+1)], 2] if k < nz - 1 else w_curr

            lap_u += get_laplacian_term(u_curr, u_k_plus_1_val, u_k_minus_1_val, (i, j, min(k+2, nz-1)), (i, j, max(k-2, 0)), dz**2, nz, k, 0, velocity, idx_to_node)
            lap_v += get_laplacian_term(v_curr, v_k_plus_1_val, v_k_minus_1_val, (i, j, min(k+2, nz-1)), (i, j, max(k-2, 0)), dz**2, nz, k, 1, velocity, idx_to_node)
            lap_w += get_laplacian_term(w_curr, w_k_plus_1_val, w_k_minus_1_val, (i, j, min(k+2, nz-1)), (i, j, max(k-2, 0)), dz**2, nz, k, 2, velocity, idx_to_node)

        diffusion_accel[node_1d_idx, 0] = viscosity * lap_u
        diffusion_accel[node_1d_idx, 1] = viscosity * lap_v
        diffusion_accel[node_1d_idx, 2] = viscosity * lap_w
    
    return diffusion_accel