import numpy as np
import math

def create_structured_grid_info(grid_dims, domain_size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Creates synthetic 3D structured grid information based on explicit grid dimensions.
    The grid is offset by the specified origin.
    Returns:
        num_nodes (int): Total number of nodes (Nx * Ny * Nz).
        nodes_coords (np.array): (num_nodes, 3) array of node coordinates.
        grid_shape (tuple): (Nx, Ny, Nz) dimensions of the grid.
        dx, dy, dz (float): Grid spacing in each dimension.
        node_to_idx (dict): Maps 1D node index to 3D (i,j,k) grid indices.
        idx_to_node (dict): Maps 3D (i,j,k) grid indices to 1D node index.
    """
    nx, ny, nz = grid_dims
    num_nodes = nx * ny * nz
    
    dx = domain_size[0] / (nx - 1) if nx > 1 else 0.0
    dy = domain_size[1] / (ny - 1) if ny > 1 else 0.0
    dz = domain_size[2] / (nz - 1) if nz > 1 else 0.0
    
    nodes_coords = np.zeros((num_nodes, 3))
    node_to_idx = {}
    idx_to_node = {}
    
    count = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes_coords[count, 0] = origin[0] + i * dx
                nodes_coords[count, 1] = origin[1] + j * dy
                nodes_coords[count, 2] = origin[2] + k * dz
                node_to_idx[count] = (i, j, k)
                idx_to_node[(i, j, k)] = count
                count += 1
    
    final_dx = domain_size[0] / (nx - 1) if nx > 1 else domain_size[0]
    final_dy = domain_size[1] / (ny - 1) if ny > 1 else domain_size[1]
    final_dz = domain_size[2] / (nz - 1) if nz > 1 else domain_size[2]

    return num_nodes, nodes_coords, grid_dims, final_dx, final_dy, final_dz, node_to_idx, idx_to_node


def find_optimal_grid_dimensions(num_nodes, domain_size, tolerance=1e-9):
    """
    Finds three integer factors (nx, ny, nz) for num_nodes such that they are as close
    to each other as possible, respecting dimensions with effectively zero or non-zero extent.
    """
    factors = []
    for i in range(1, int(math.sqrt(num_nodes)) + 1):
        if num_nodes % i == 0:
            factors.append(i)
            if i * i != num_nodes:
                factors.append(num_nodes // i)
    factors.sort()

    best_dims = (1, 1, num_nodes)
    min_diff = float('inf')

    target_dim_size = num_nodes**(1/3)

    for nz_val in factors:
        if num_nodes % nz_val == 0:
            remaining_nodes = num_nodes // nz_val
            for nx_val in factors:
                if remaining_nodes % nx_val == 0:
                    ny_val = remaining_nodes // nx_val

                    if nx_val * ny_val * nz_val != num_nodes:
                        continue

                    current_dims = tuple(sorted((nx_val, ny_val, nz_val)))

                    is_valid_combination = True
                    for dim_idx in range(3):
                        if domain_size[dim_idx] > tolerance and current_dims[dim_idx] == 1:
                            is_valid_combination = False
                            break
                        if domain_size[dim_idx] < tolerance and current_dims[dim_idx] > 1:
                            is_valid_combination = False
                            break
                    
                    if not is_valid_combination:
                        continue

                    diff_score = abs(nx_val - target_dim_size) + \
                                 abs(ny_val - target_dim_size) + \
                                 abs(nz_val - target_dim_size)

                    if diff_score < min_diff:
                        min_diff = diff_score
                        best_dims = current_dims
    
    return best_dims if all(d >= 1 for d in best_dims) else (1,1,num_nodes)