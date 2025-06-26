# src/utils/grid.py

import numpy as np
import sys  # For stderr

def create_structured_grid_info(domain_definition: dict) -> dict:
    """
    Creates detailed mesh information for a structured grid from a domain_definition dictionary.
    """
    min_x, max_x = domain_definition['min_x'], domain_definition['max_x']
    min_y, max_y = domain_definition['min_y'], domain_definition['max_y']
    min_z, max_z = domain_definition['min_z'], domain_definition['max_z']
    nx, ny, nz = domain_definition['nx'], domain_definition['ny'], domain_definition['nz']

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid dimensions: nx={nx}, ny={ny}, nz={nz}. Must be positive integers.")

    TOLERANCE = 1e-9
    dx = (max_x - min_x) / nx if (max_x - min_x) > TOLERANCE else 1.0
    dy = (max_y - min_y) / ny if (max_y - min_y) > TOLERANCE else 1.0
    dz = (max_z - min_z) / nz if (max_z - min_z) > TOLERANCE else 1.0

    num_cells = nx * ny * nz
    cell_centers = get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz)

    return {
        'nx': nx, 'ny': ny, 'nz': nz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'min_z': min_z, 'max_z': max_z,
        'num_cells': num_cells,
        'cell_centers': cell_centers
    }

def get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz):
    """
    Generates cell center coordinates for a uniform structured grid.
    """
    x_edges = np.linspace(min_x, max_x, nx + 1)
    y_edges = np.linspace(min_y, max_y, ny + 1)
    z_edges = np.linspace(min_z, max_z, nz + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    Xc, Yc, Zc = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    return np.vstack([Xc.ravel(), Yc.ravel(), Zc.ravel()]).T

def create_mac_grid_fields(grid_shape, ghost_width=1):
    """
    Creates velocity and pressure arrays for a MAC grid with ghost cells.

    Args:
        grid_shape (tuple): Base shape (nx, ny, nz)
        ghost_width (int): Number of ghost cells on each boundary

    Returns:
        dict of ndarray: Keys are 'u', 'v', 'w', and 'p'
    """
    nx, ny, nz = grid_shape
    gx = ghost_width

    fields = {
        "u": np.zeros((nx + 1 + 2 * gx, ny + 2 * gx,     nz + 2 * gx), dtype=np.float64),
        "v": np.zeros((nx + 2 * gx,     ny + 1 + 2 * gx, nz + 2 * gx), dtype=np.float64),
        "w": np.zeros((nx + 2 * gx,     ny + 2 * gx,     nz + 1 + 2 * gx), dtype=np.float64),
        "p": np.zeros((nx + 2 * gx,     ny + 2 * gx,     nz + 2 * gx), dtype=np.float64)
    }
    return fields

def generate_physical_coordinates(grid_shape, spacing, origin=(0.0, 0.0, 0.0), ghost_width=1):
    """
    Computes physical coordinates of staggered field centers (u, v, w, p).

    Returns:
        dict of (x, y, z) arrays for each field.
    """
    nx, ny, nz = grid_shape
    dx, dy, dz = spacing
    ox, oy, oz = origin
    gx = ghost_width

    def lin(start, count, delta, offset):
        return start + (np.arange(count) - gx + offset) * delta

    coords = {
        "u": (lin(ox, nx + 1 + 2 * gx, dx, 0.0),
              lin(oy, ny     + 2 * gx, dy, 0.5),
              lin(oz, nz     + 2 * gx, dz, 0.5)),

        "v": (lin(ox, nx     + 2 * gx, dx, 0.5),
              lin(oy, ny + 1 + 2 * gx, dy, 0.0),
              lin(oz, nz     + 2 * gx, dz, 0.5)),

        "w": (lin(ox, nx     + 2 * gx, dx, 0.5),
              lin(oy, ny     + 2 * gx, dy, 0.5),
              lin(oz, nz + 1 + 2 * gx, dz, 0.0)),

        "p": (lin(ox, nx     + 2 * gx, dx, 0.5),
              lin(oy, ny     + 2 * gx, dy, 0.5),
              lin(oz, nz     + 2 * gx, dz, 0.5))
    }
    return coords
