import numpy as np
import math

def create_structured_grid_info(x_coords_grid_lines, y_coords_grid_lines, z_coords_grid_lines):
    """
    Creates 3D structured grid information based on explicit lists of sorted unique
    coordinates for each dimension. This directly constructs the grid points from
    the inferred grid lines, ensuring perfect alignment with the input's spatial definition.

    Args:
        x_coords_grid_lines (list): Sorted list of unique x-coordinates defining grid lines.
        y_coords_grid_lines (list): Sorted list of unique y-coordinates defining grid lines.
        z_coords_grid_lines (list): Sorted list of unique z-coordinates defining grid lines.

    Returns:
        num_nodes (int): Total number of nodes (Nx * Ny * Nz).
        nodes_coords (np.array): (num_nodes, 3) array of node coordinates.
        grid_shape (tuple): (Nx, Ny, Nz) dimensions of the grid.
        dx, dy, dz (float): Grid spacing in each dimension.
        node_to_idx (dict): Maps 1D node index to 3D (i,j,k) grid indices.
        idx_to_node (dict): Maps 3D (i,j,k) grid indices to 1D node index.
    """
    nx = len(x_coords_grid_lines)
    ny = len(y_coords_grid_lines)
    nz = len(z_coords_grid_lines)
    
    grid_shape = (nx, ny, nz)
    num_nodes = nx * ny * nz
    
    # Calculate dx, dy, dz based on the spacing between the first two grid lines,
    # or 0.0 if there's only one grid line in that dimension.
    # Handle the case where there's only one node in a dimension (e.g., a 2D problem)
    dx = x_coords_grid_lines[1] - x_coords_grid_lines[0] if nx > 1 else 0.0
    dy = y_coords_grid_lines[1] - y_coords_grid_lines[0] if ny > 1 else 0.0
    dz = z_coords_grid_lines[1] - z_coords_grid_lines[0] if nz > 1 else 0.0

    nodes_coords = np.zeros((num_nodes, 3))
    node_to_idx = {}
    idx_to_node = {}
    
    count = 0
    # Iterate in Z, then Y, then X for standard 1D indexing mapping (k, j, i)
    for k_idx in range(nz):
        for j_idx in range(ny):
            for i_idx in range(nx):
                # Populate coordinates directly from the provided grid line lists
                nodes_coords[count, 0] = x_coords_grid_lines[i_idx]
                nodes_coords[count, 1] = y_coords_grid_lines[j_idx]
                nodes_coords[count, 2] = z_coords_grid_lines[k_idx]
                
                node_to_idx[count] = (i_idx, j_idx, k_idx)
                idx_to_node[(i_idx, j_idx, k_idx)] = count
                count += 1
    
    return num_nodes, nodes_coords, grid_shape, dx, dy, dz, node_to_idx, idx_to_node


def find_optimal_grid_dimensions(num_nodes, domain_size, tolerance=1e-9):
    """
    Finds three integer factors (nx, ny, nz) for num_nodes such that they are as close
    to the aspect ratio of the domain as possible.
    Prioritizes dimensions with non-zero extent, ensuring single nodes for zero-extent dimensions.
    """
    
    # Determine which dimensions have non-zero extent
    has_x_extent = domain_size[0] > tolerance
    has_y_extent = domain_size[1] > tolerance
    has_z_extent = domain_size[2] > tolerance

    # If any dimension has zero extent, it must have only 1 node in the grid_dims
    # If all extents are zero, it's a single point, so (1,1,1)
    if not has_x_extent and not has_y_extent and not has_z_extent:
        return (1, 1, 1) if num_nodes == 1 else (num_nodes, 1, 1) # Or raise error if num_nodes > 1

    # Get all factors of num_nodes
    factors = []
    for i in range(1, int(math.sqrt(num_nodes)) + 1):
        if num_nodes % i == 0:
            factors.append(i)
            if i * i != num_nodes:
                factors.append(num_nodes // i)
    factors.sort()

    best_dims = None
    min_aspect_ratio_diff = float('inf')

    # Iterate through all possible combinations of three factors
    # This loop assumes a preference for (nx, ny, nz) order matching (x, y, z) domain_size
    # We prioritize finding factors that align with the non-zero dimensions.
    
    # Start with a base case for 1D/2D scenarios
    # These base cases should ideally be caught by the direct coordinate inference in main_solver.py,
    # but kept here for robustness of this function.
    if not has_y_extent and not has_z_extent: # 1D along X
        return (num_nodes, 1, 1)
    if not has_x_extent and not has_z_extent: # 1D along Y
        return (1, num_nodes, 1)
    if not has_x_extent and not has_y_extent: # 1D along Z
        return (1, 1, num_nodes)
    
    # Consider 2D cases
    if not has_z_extent: # 2D in XY plane
        for nx_val in factors:
            if num_nodes % nx_val == 0:
                ny_val = num_nodes // nx_val
                # Check for other dimensions being 1
                if ny_val * nx_val == num_nodes:
                    current_dims = (nx_val, ny_val, 1)
                    if current_dims[0] == 1 and not has_x_extent: continue # Skip if X has no extent but nx > 1
                    if current_dims[1] == 1 and not has_y_extent: continue # Skip if Y has no extent but ny > 1

                    if has_x_extent and has_y_extent:
                        target_aspect_ratio = domain_size[0] / domain_size[1]
                        current_aspect_ratio = nx_val / ny_val if ny_val > 0 else float('inf')
                        diff = abs(np.log(current_aspect_ratio) - np.log(target_aspect_ratio))
                    else: # One of the dimensions has zero extent, so that dimension must be 1 node
                        diff = 0 

                    if diff < min_aspect_ratio_diff:
                        min_aspect_ratio_diff = diff
                        best_dims = current_dims
        if best_dims: return best_dims

    if not has_y_extent: # 2D in XZ plane
        for nx_val in factors:
            if num_nodes % nx_val == 0:
                nz_val = num_nodes // nx_val
                if nz_val * nx_val == num_nodes:
                    current_dims = (nx_val, 1, nz_val)
                    if current_dims[0] == 1 and not has_x_extent: continue
                    if current_dims[2] == 1 and not has_z_extent: continue
                    
                    if has_x_extent and has_z_extent:
                        target_aspect_ratio = domain_size[0] / domain_size[2]
                        current_aspect_ratio = nx_val / nz_val if nz_val > 0 else float('inf')
                        diff = abs(np.log(current_aspect_ratio) - np.log(target_aspect_ratio))
                    else:
                        diff = 0

                    if diff < min_aspect_ratio_diff:
                        min_aspect_ratio_diff = diff
                        best_dims = current_dims
        if best_dims: return best_dims
    
    if not has_x_extent: # 2D in YZ plane
        for ny_val in factors:
            if num_nodes % ny_val == 0:
                nz_val = num_nodes // ny_val
                if nz_val * ny_val == num_nodes:
                    current_dims = (1, ny_val, nz_val)
                    if current_dims[1] == 1 and not has_y_extent: continue
                    if current_dims[2] == 1 and not has_z_extent: continue
                    
                    if has_y_extent and has_z_extent:
                        target_aspect_ratio = domain_size[1] / domain_size[2]
                        current_aspect_ratio = ny_val / nz_val if nz_val > 0 else float('inf')
                        diff = abs(np.log(current_aspect_ratio) - np.log(target_aspect_ratio))
                    else:
                        diff = 0

                    if diff < min_aspect_ratio_diff:
                        min_aspect_ratio_diff = diff
                        best_dims = current_dims
        if best_dims: return best_dims


    # If it's a full 3D case (all extents > 0) or fallback
    min_diff_from_cube_root = float('inf') # This variable name is a bit misleading now, as it's min_aspect_ratio_diff
    best_cube_root_dims = (1, 1, num_nodes) # Fallback

    # Try to find dimensions that respect the domain_size aspect ratio
    for nx_val in factors:
        if num_nodes % nx_val == 0:
            remaining = num_nodes // nx_val
            for ny_val in factors:
                if remaining % ny_val == 0:
                    nz_val = remaining // ny_val

                    if nx_val * ny_val * nz_val != num_nodes:
                        continue # Should not happen with correct factor finding

                    # Check if the dimensions are compatible with the domain extent
                    # If domain_size[dim_idx] is close to zero, current_dims[dim_idx] MUST be 1
                    if (not has_x_extent and nx_val > 1) or \
                       (not has_y_extent and ny_val > 1) or \
                       (not has_z_extent and nz_val > 1):
                        continue
                    
                    # If domain_size[dim_idx] is significant, current_dims[dim_idx] MUST be > 1 (unless num_nodes is 1)
                    if (has_x_extent and nx_val == 1 and num_nodes > 1) or \
                       (has_y_extent and ny_val == 1 and num_nodes > 1) or \
                       (has_z_extent and nz_val == 1 and num_nodes > 1):
                        continue
                    
                    current_dims = (nx_val, ny_val, nz_val)

                    # Calculate a score based on matching the domain aspect ratio
                    score = 0.0
                    for i in range(3):
                        if domain_size[i] > tolerance: # Only consider dimensions with extent
                            if current_dims[i] > 1:
                                # The ideal "cell size" in this dimension
                                score += (domain_size[i] / (current_dims[i] - 1)) ** 2
                            elif domain_size[i] > tolerance: # If it has extent but only 1 node, penalize
                                score += (domain_size[i] * 100)**2 # Large penalty for not dividing if it has extent
                        elif current_dims[i] > 1: # If domain has no extent, but grid dim > 1, penalize
                            score += 1000 # Very large penalty
                    
                    if score < min_aspect_ratio_diff:
                        min_aspect_ratio_diff = score
                        best_dims = current_dims

    # Fallback if no valid combination is found, or for single node cases
    if best_dims is None:
        if num_nodes == 1:
            return (1, 1, 1)
        elif has_x_extent and not has_y_extent and not has_z_extent:
            return (num_nodes, 1, 1)
        elif not has_x_extent and has_y_extent and not has_z_extent:
            return (1, num_nodes, 1)
        elif not has_x_extent and not has_y_extent and has_z_extent:
            return (1, 1, num_nodes)
        else: # Generic 3D fallback (try to make it cubic)
            nx = int(round(num_nodes**(1/3)))
            while num_nodes % nx != 0:
                nx -= 1
            if nx == 0: nx = 1 # Avoid division by zero for small num_nodes
            remaining = num_nodes // nx
            ny = int(round(remaining**(1/2)))
            while remaining % ny != 0:
                ny -= 1
            if ny == 0: ny = 1 # Avoid division by zero
            nz = remaining // ny
            return (nx, ny, nz)
    
    return best_dims
