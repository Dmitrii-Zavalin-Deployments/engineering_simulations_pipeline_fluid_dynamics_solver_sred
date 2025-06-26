# tests/test_grid_initialization/test_field_location_mapping.py

import numpy as np
import pytest

# 🔧 Adjust this to your actual grid builder
from src.grid.grid_builder import create_mac_grid_fields, generate_physical_coordinates

def test_field_center_coordinates_match_expected_offsets():
    """
    Verifies that the physical coordinates of u, v, w, and p fields
    match MAC grid staggering relative to the origin and spacing.
    """
    nx, ny, nz = 4, 3, 2
    ghost = 1
    dx = dy = dz = 1.0
    origin = (0.0, 0.0, 0.0)

    # Call real grid logic to generate fields
    fields = create_mac_grid_fields((nx, ny, nz), ghost_width=ghost)
    coords = generate_physical_coordinates((nx, ny, nz), (dx, dy, dz), origin, ghost_width=ghost)

    # Unpack expected coordinates
    u_x, u_y, u_z = coords["u"]
    v_x, v_y, v_z = coords["v"]
    w_x, w_y, w_z = coords["w"]
    p_x, p_y, p_z = coords["p"]

    # Validate staggering: u is offset by 0.0 in x, but 0.5 dx in y/z, etc.
    assert np.isclose(u_x[0], origin[0]), "u_x should start at domain origin"
    assert np.isclose(u_y[0], origin[1] + 0.5 * dy), "u_y not properly staggered"
    assert np.isclose(v_x[0], origin[0] + 0.5 * dx), "v_x not properly staggered"
    assert np.isclose(w_z[0], origin[2], "w_z should start at domain origin")

    # Validate cell-centered pressure offset
    assert np.isclose(p_x[0], origin[0] + 0.5 * dx), "p_x should be at cell centers"
    assert np.isclose(p_y[0], origin[1] + 0.5 * dy), "p_y should be at cell centers"
    assert np.isclose(p_z[0], origin[2] + 0.5 * dz), "p_z should be at cell centers"

    # Ensure full coverage
    assert len(p_x) == nx, "Unexpected number of pressure points in x"
    assert len(u_x) == nx + 1, "Unexpected number of u points in x"
    assert len(v_y) == ny + 1, "Unexpected number of v points in y"
    assert len(w_z) == nz + 1, "Unexpected number of w points in z"



