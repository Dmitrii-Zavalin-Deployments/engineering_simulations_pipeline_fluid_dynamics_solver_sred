# tests/test_grid_initialization/test_field_location_mapping.py

import numpy as np
import pytest
from src.utils.grid import create_mac_grid_fields, generate_physical_coordinates

def test_field_center_coordinates_match_expected_offsets():
    """
    Verifies that physical coordinates of u, v, w, and p match expected staggering for MAC grid.
    """
    nx, ny, nz = 4, 3, 2
    ghost = 1
    dx = dy = dz = 1.0
    origin = (0.0, 0.0, 0.0)

    coords = generate_physical_coordinates((nx, ny, nz), (dx, dy, dz), origin, ghost_width=ghost)

    # Expected first coordinates (after ghost offsets)
    assert np.isclose(coords["u"][0][0], origin[0] - dx * ghost), "u_x start offset mismatch"
    assert np.isclose(coords["u"][1][0], origin[1] - dy * ghost + 0.5 * dy), "u_y staggering mismatch"
    assert np.isclose(coords["u"][2][0], origin[2] - dz * ghost + 0.5 * dz), "u_z staggering mismatch"

    assert np.isclose(coords["v"][0][0], origin[0] - dx * ghost + 0.5 * dx), "v_x staggering mismatch"
    assert np.isclose(coords["v"][1][0], origin[1] - dy * ghost), "v_y start offset mismatch"
    assert np.isclose(coords["v"][2][0], origin[2] - dz * ghost + 0.5 * dz), "v_z staggering mismatch"

    assert np.isclose(coords["w"][0][0], origin[0] - dx * ghost + 0.5 * dx), "w_x staggering mismatch"
    assert np.isclose(coords["w"][1][0], origin[1] - dy * ghost + 0.5 * dy), "w_y staggering mismatch"
    assert np.isclose(coords["w"][2][0], origin[2] - dz * ghost), "w_z start offset mismatch"

    assert np.isclose(coords["p"][0][0], origin[0] - dx * ghost + 0.5 * dx), "p_x staggering mismatch"
    assert np.isclose(coords["p"][1][0], origin[1] - dy * ghost + 0.5 * dy), "p_y staggering mismatch"
    assert np.isclose(coords["p"][2][0], origin[2] - dz * ghost + 0.5 * dz), "p_z staggering mismatch"



