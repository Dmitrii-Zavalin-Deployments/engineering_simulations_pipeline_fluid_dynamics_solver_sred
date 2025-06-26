# tests/test_grid_initialization/test_ghost_cell_indexing.py

import numpy as np
import pytest

def create_padded_grid(shape, padding=1):
    """
    Returns a padded grid with ghost cells around the computational domain.
    The core grid values are set to 1, ghost cells to -1.
    """
    nx, ny, nz = shape
    padded_shape = (nx + 2 * padding, ny + 2 * padding, nz + 2 * padding)
    grid = np.full(padded_shape, fill_value=-1.0)  # Ghost cells
    grid[padding:-padding, padding:-padding, padding:-padding] = 1.0  # Real domain
    return grid

def test_ghost_cell_layer_surrounds_domain():
    """
    Verify that a 1-cell-wide ghost layer wraps the real domain on all faces.
    """
    nx, ny, nz = 4, 3, 2
    ghost_width = 1
    grid = create_padded_grid((nx, ny, nz), padding=ghost_width)

    # Check interior values (should be 1.0)
    interior = grid[ghost_width:-ghost_width, ghost_width:-ghost_width, ghost_width:-ghost_width]
    assert np.all(interior == 1.0), "Interior domain values incorrect."

    # Check each ghost face for -1.0
    assert np.all(grid[0, :, :] == -1.0), "Ghost cells on min x-face not set correctly."
    assert np.all(grid[-1, :, :] == -1.0), "Ghost cells on max x-face not set correctly."
    assert np.all(grid[:, 0, :] == -1.0), "Ghost cells on min y-face not set correctly."
    assert np.all(grid[:, -1, :] == -1.0), "Ghost cells on max y-face not set correctly."
    assert np.all(grid[:, :, 0] == -1.0), "Ghost cells on min z-face not set correctly."
    assert np.all(grid[:, :, -1] == -1.0), "Ghost cells on max z-face not set correctly."



