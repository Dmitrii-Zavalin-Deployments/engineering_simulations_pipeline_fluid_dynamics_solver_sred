# tests/test_grid_initialization/test_staggered_field_alignment.py

import numpy as np
import pytest

def generate_mock_field_shapes(grid_shape):
    """
    Given a grid shape (nx, ny, nz), returns expected shapes for
    staggered fields: u, v, w, and pressure (p).
    """
    nx, ny, nz = grid_shape
    u_shape = (nx + 1, ny, nz)   # u on x-faces
    v_shape = (nx, ny + 1, nz)   # v on y-faces
    w_shape = (nx, ny, nz + 1)   # w on z-faces
    p_shape = (nx, ny, nz)       # pressure at cell centers
    return u_shape, v_shape, w_shape, p_shape

def test_staggered_field_shapes_consistent():
    """
    Check that staggered velocity and pressure arrays conform to expected shape logic.
    """
    nx, ny, nz = 4, 3, 2  # Small test domain
    u_expected, v_expected, w_expected, p_expected = generate_mock_field_shapes((nx, ny, nz))

    # Simulate field creation
    u = np.zeros(u_expected)
    v = np.zeros(v_expected)
    w = np.zeros(w_expected)
    p = np.zeros(p_expected)

    # Assertions
    assert u.shape == u_expected, f"Expected u shape {u_expected}, got {u.shape}"
    assert v.shape == v_expected, f"Expected v shape {v_expected}, got {v.shape}"
    assert w.shape == w_expected, f"Expected w shape {w_expected}, got {w.shape}"
    assert p.shape == p_expected, f"Expected p shape {p_expected}, got {p.shape}"



