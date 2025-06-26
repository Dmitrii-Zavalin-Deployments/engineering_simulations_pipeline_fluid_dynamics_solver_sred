# tests/test_grid_initialization/test_structured_grid.py

import pytest
import numpy as np
from src.utils.grid import create_structured_grid_info, get_cell_centers

def test_grid_info_calculates_correct_spacing():
    domain = {
        'min_x': 0.0, 'max_x': 4.0,
        'min_y': 0.0, 'max_y': 2.0,
        'min_z': 0.0, 'max_z': 1.0,
        'nx': 4, 'ny': 2, 'nz': 1
    }

    info = create_structured_grid_info(domain)

    assert info['dx'] == 1.0
    assert info['dy'] == 1.0
    assert info['dz'] == 1.0
    assert info['num_cells'] == 4 * 2 * 1
    assert info['cell_centers'].shape == (8, 3)

def test_grid_info_fallback_spacing_when_extent_is_zero():
    domain = {
        'min_x': 0.0, 'max_x': 0.0,  # no extent
        'min_y': 0.0, 'max_y': 2.0,
        'min_z': 0.0, 'max_z': 2.0,
        'nx': 1, 'ny': 2, 'nz': 2
    }

    info = create_structured_grid_info(domain)
    assert info['dx'] == 1.0  # fallback default
    assert info['dy'] == 1.0
    assert info['dz'] == 1.0
    assert info['cell_centers'].shape == (1 * 2 * 2, 3)

def test_invalid_cell_counts_raise_exception():
    domain = {
        'min_x': 0, 'max_x': 1,
        'min_y': 0, 'max_y': 1,
        'min_z': 0, 'max_z': 1,
        'nx': -1, 'ny': 2, 'nz': 2
    }
    with pytest.raises(ValueError, match="Invalid grid dimensions"):
        create_structured_grid_info(domain)

def test_get_cell_centers_values_are_within_bounds():
    min_x, max_x = 0.0, 2.0
    min_y, max_y = -1.0, 1.0
    min_z, max_z = 10.0, 12.0
    nx, ny, nz = 2, 2, 2

    centers = get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz)
    assert centers.shape == (nx * ny * nz, 3)

    x_min_allowed = min_x + 0.5 * (max_x - min_x) / nx
    x_max_allowed = max_x - 0.5 * (max_x - min_x) / nx
    y_min_allowed = min_y + 0.5 * (max_y - min_y) / ny
    y_max_allowed = max_y - 0.5 * (max_y - min_y) / ny
    z_min_allowed = min_z + 0.5 * (max_z - min_z) / nz
    z_max_allowed = max_z - 0.5 * (max_z - min_z) / nz

    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    assert np.all((x >= x_min_allowed) & (x <= x_max_allowed))
    assert np.all((y >= y_min_allowed) & (y <= y_max_allowed))
    assert np.all((z >= z_min_allowed) & (z <= z_max_allowed))



