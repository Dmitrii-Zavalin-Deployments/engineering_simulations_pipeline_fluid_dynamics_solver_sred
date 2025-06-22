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
    # If a dimension has only one grid line (e.g., a 2D problem embedded in 3D),
    # nx, ny, or nz should be 1, and dx, dy, dz handled appropriately.
    nx_raw = len(x_coords) - 1
    ny_raw = len(y_coords) - 1
    nz_raw = len(z_coords) - 1

    # Ensure at least one cell if only one unique coordinate is present
    # This implies a 2D or 1D effective domain.
    nx = max(1, nx_raw)
    ny = max(1, ny_raw)
    nz = max(1, nz_raw)

    grid_shape = (nx, ny, nz)
    num_nodes = nx * ny * nz

    # Calculate cell center coordinates and grid spacing
    # Handle cases where a dimension effectively collapses (e.g., nx_raw <= 0)
    dx = (x_coords[1] - x_coords[0]) if nx_raw > 0 else 1.0 # Nominal 1.0 if collapsed
    dy = (y_coords[1] - y_coords[0]) if ny_raw > 0 else 1.0
    dz = (z_coords[1] - z_coords[0]) if nz_raw > 0 else 1.0

    if nx_raw > 0 and not np.allclose(np.diff(x_coords), dx):
        print("Warning: Non-uniform grid spacing detected in X. Using first interval for dx.", file=sys.stderr)
    if ny_raw > 0 and not np.allclose(np.diff(y_coords), dy):
        print("Warning: Non-uniform grid spacing detected in Y. Using first interval for dy.", file=sys.stderr)
    if nz_raw > 0 and not np.allclose(np.diff(z_coords), dz):
        print("Warning: Non-uniform grid spacing detected in Z. Using first interval for dz.", file=sys.stderr)

    # Generate cell center coordinates (for `nodes_coords` for compatibility/debug)
    # Cell center is (x_i + x_{i+1})/2
    cell_centers_x = (x_coords[:-1] + x_coords[1:]) / 2 if nx_raw > 0 else np.array([(x_coords[0] + x_coords[0])/2]) # If nx=1, center is just the point
    cell_centers_y = (y_coords[:-1] + y_coords[1:]) / 2 if ny_raw > 0 else np.array([(y_coords[0] + y_coords[0])/2])
    cell_centers_z = (z_coords[:-1] + z_coords[1:]) / 2 if nz_raw > 0 else np.array([(z_coords[0] + z_coords[0])/2])

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

def get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz):
    """
    Generates cell center coordinates for a uniform structured grid based on min/max extents and cell counts.
    This is suitable for cases where the grid is inferred from overall domain dimensions.

    Args:
        min_x, max_x (float): Min/max coordinates for X-axis.
        min_y, max_y (float): Min/max coordinates for Y-axis.
        min_z, max_z (float): Min/max coordinates for Z-axis.
        nx, ny, nz (int): Number of cells in X, Y, Z directions.

    Returns:
        np.ndarray: A (nx*ny*nz, 3) array of cell center coordinates.
    """
    # Calculate cell edge coordinates
    # If nx is 1, linspace will give [min_x, max_x]. We need to ensure a single center point.
    # The (nx + 1) is for the number of grid lines/edges.
    x_edges = np.linspace(min_x, max_x, nx + 1)
    y_edges = np.linspace(min_y, max_y, ny + 1)
    z_edges = np.linspace(min_z, max_z, nz + 1)

    # Calculate cell center coordinates by averaging adjacent edges
    # If nx=1, x_edges is [min_x, max_x], x_centers will be [(min_x+max_x)/2]. This is correct.
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    # Create a meshgrid for cell center coordinates
    X_centers, Y_centers, Z_centers = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    
    # Flatten and combine into a (num_nodes, 3) array
    nodes_coords = np.vstack([X_centers.ravel(), Y_centers.ravel(), Z_centers.ravel()]).T
    
    return nodes_coords
