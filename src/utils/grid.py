# src/utils/grid.py

import numpy as np
import sys # Import sys for stderr in warnings

def create_structured_grid_info(domain_definition: dict) -> dict:
    """
    Creates detailed mesh information for a structured grid from a domain_definition dictionary.
    This function calculates cell dimensions (nx, ny, nz), grid spacing (dx, dy, dz),
    domain extents, and cell center coordinates.

    Args:
        domain_definition (dict): A dictionary containing:
                                  - 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z' (float)
                                  - 'nx', 'ny', 'nz' (int) representing number of cells.

    Returns:
        dict: A dictionary containing:
              - 'nx', 'ny', 'nz': Number of cells in each dimension.
              - 'dx', 'dy', 'dz': Grid spacing in each dimension.
              - 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z': Domain extents.
              - 'num_cells': Total number of grid cells.
              - 'cell_centers': (num_cells, 3) numpy array of cell center coordinates (flattened).
    """
    min_x, max_x = domain_definition['min_x'], domain_definition['max_x']
    min_y, max_y = domain_definition['min_y'], domain_definition['max_y']
    min_z, max_z = domain_definition['min_z'], domain_definition['max_z']
    nx, ny, nz = domain_definition['nx'], domain_definition['ny'], domain_definition['nz']

    # Validate dimensions: Ensure positive cell counts
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid dimensions: nx={nx}, ny={ny}, nz={nz}. Must be positive integers.")

    # Calculate grid spacing (dx, dy, dz)
    # Handle cases where a dimension might effectively collapse (e.g., nx=1)
    # and min_val == max_val in the input. In such cases, dx should not be 0.
    # Use a small epsilon for floating point comparison to determine effective extent
    TOLERANCE = 1e-9

    dx = (max_x - min_x) / nx if nx > 0 and (max_x - min_x) > TOLERANCE else 1.0
    dy = (max_y - min_y) / ny if ny > 0 and (max_y - min_y) > TOLERANCE else 1.0
    dz = (max_z - min_z) / nz if nz > 0 and (max_z - min_z) > TOLERANCE else 1.0

    # For 1-cell dimensions where min_val == max_val, spacing is set to 1.0.
    # This ensures consistency for solvers and VTK visualization, though physically
    # it means the dimension has no extent.

    num_cells = nx * ny * nz

    # Generate cell center coordinates using the existing get_cell_centers function
    cell_centers = get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz)

    # Prepare the mesh_info dictionary to be returned
    mesh_info = {
        'nx': nx, 'ny': ny, 'nz': nz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'min_z': min_z, 'max_z': max_z,
        'num_cells': num_cells,
        'cell_centers': cell_centers # This is a numpy array
    }
    return mesh_info

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
    # The (N + 1) is for the number of grid lines/edges.
    x_edges = np.linspace(min_x, max_x, nx + 1)
    y_edges = np.linspace(min_y, max_y, ny + 1)
    z_edges = np.linspace(min_z, max_z, nz + 1)

    # Calculate cell center coordinates by averaging adjacent edges
    # If nx=1, x_edges is [min_x, max_x], x_centers will be [(min_x+max_x)/2]. This is correct.
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    # Create a meshgrid for cell center coordinates
    # Use 'ij' indexing for (nx, ny, nz) grid shapes
    X_centers, Y_centers, Z_centers = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    # Flatten and combine into a (num_nodes, 3) array
    nodes_coords = np.vstack([X_centers.ravel(), Y_centers.ravel(), Z_centers.ravel()]).T

    return nodes_coords
