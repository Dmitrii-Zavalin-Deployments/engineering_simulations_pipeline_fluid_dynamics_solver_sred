import numpy as np

def create_structured_grid_info(x_grid_lines_coords, y_grid_lines_coords, z_grid_lines_coords):
    """
    Creates detailed mesh information for a structured grid from ordered coordinate arrays.
    This function processes the unique coordinates to define grid cells and spacing.

    Args:
        x_grid_lines_coords (list/np.ndarray): Sorted list/array of unique X coordinates defining cell faces.
        y_grid_lines_coords (list/np.ndarray): Sorted list/array of unique Y coordinates defining cell faces.
        z_grid_lines_coords (list/np.ndarray): Sorted list/array of unique Z coordinates defining cell faces.

    Returns:
        tuple:
            num_nodes (int): Total number of grid cells/nodes (nx * ny * nz).
            nodes_coords (np.ndarray): A (num_nodes, 3) array of cell center coordinates.
                                      (Kept for compatibility/debug, not used in core vectorized calcs)
            grid_shape (tuple): (nx, ny, nz) representing the number of cells in each dimension.
            dx (float): Grid spacing in X direction (assumed uniform for now, picks first).
            dy (float): Grid spacing in Y direction (assumed uniform for now, picks first).
            dz (float): Grid spacing in Z direction (assumed uniform for now, picks first).
            node_to_idx (dict): Maps (i, j, k) tuple to linear index.
                                (Kept for compatibility/debug, not used in core vectorized calcs)
            idx_to_node (dict): Maps linear index to (i, j, k) tuple.
                                (Kept for compatibility/debug, not used in core vectorized calcs)
    """
    # Convert to NumPy arrays and ensure unique sorted values
    x_coords = np.array(sorted(list(set(x_grid_lines_coords))))
    y_coords = np.array(sorted(list(set(y_grid_lines_coords))))
    z_coords = np.array(sorted(list(set(z_grid_lines_coords))))

    # Calculate number of cells in each dimension
    # Number of cells is (number of grid lines - 1)
    nx = len(x_coords) - 1
    ny = len(y_coords) - 1
    nz = len(z_coords) - 1

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Grid dimensions must be at least 1x1x1. Ensure at least two unique grid line coordinates per dimension.")

    grid_shape = (nx, ny, nz)
    num_nodes = nx * ny * nz

    # Calculate cell center coordinates and grid spacing
    # Assuming uniform spacing for now, taking the first interval.
    # If non-uniform, dx, dy, dz would need to be arrays.
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    dz = z_coords[1] - z_coords[0]

    if not np.allclose(np.diff(x_coords), dx) or \
       not np.allclose(np.diff(y_coords), dy) or \
       not np.allclose(np.diff(z_coords), dz):
        print("Warning: Non-uniform grid spacing detected. Using first interval for dx, dy, dz.")
        # For a truly non-uniform grid, dx, dy, dz would need to be arrays of cell sizes
        # and numerical methods would need to adapt. For this project, we'll assume uniformity for simplicity.

    # Generate cell center coordinates (for `nodes_coords` for compatibility/debug)
    # Cell center is (x_i + x_{i+1})/2
    cell_centers_x = (x_coords[:-1] + x_coords[1:]) / 2
    cell_centers_y = (y_coords[:-1] + y_coords[1:]) / 2
    cell_centers_z = (z_coords[:-1] + z_coords[1:]) / 2

    # Create a meshgrid for cell center coordinates
    X_centers, Y_centers, Z_centers = np.meshgrid(cell_centers_x, cell_centers_y, cell_centers_z, indexing='ij')

    # Flatten and combine into a (num_nodes, 3) array
    nodes_coords = np.vstack([X_centers.ravel(), Y_centers.ravel(), Z_centers.ravel()]).T

    # Create node_to_idx and idx_to_node mappings (kept for compatibility/debug)
    node_to_idx = {}
    idx_to_node = {}
    idx_counter = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                node_to_idx[(i, j, k)] = idx_counter
                idx_to_node[idx_counter] = (i, j, k)
                idx_counter += 1

    return num_nodes, nodes_coords, grid_shape, dx, dy, dz, node_to_idx, idx_to_node