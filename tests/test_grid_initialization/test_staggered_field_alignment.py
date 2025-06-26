# tests/test_grid_initialization/test_staggered_field_alignment.py

import pytest
import numpy as np

# 🔧 Adjust this import path to your actual initialization module
from src.grid.grid_builder import create_mac_grid_fields

def test_mac_field_shapes_match_expected():
    """
    Calls the real grid initializer and confirms that staggered velocity and pressure
    fields are shaped consistently with MAC layout.
    """
    nx, ny, nz = 5, 4, 3
    ghost = 1  # Number of ghost cells applied in each direction

    # Call your actual initialization code
    fields = create_mac_grid_fields((nx, ny, nz), ghost_width=ghost)

    # Compute expected shapes with ghost zones applied
    u_shape = (nx + 1 + 2 * ghost, ny + 2 * ghost,     nz + 2 * ghost)     # u on x-faces
    v_shape = (nx + 2 * ghost,     ny + 1 + 2 * ghost, nz + 2 * ghost)     # v on y-faces
    w_shape = (nx + 2 * ghost,     ny + 2 * ghost,     nz + 1 + 2 * ghost) # w on z-faces
    p_shape = (nx + 2 * ghost,     ny + 2 * ghost,     nz + 2 * ghost)     # p at cell centers

    assert fields["u"].shape == u_shape, f"Expected u shape {u_shape}, got {fields['u'].shape}"
    assert fields["v"].shape == v_shape, f"Expected v shape {v_shape}, got {fields['v'].shape}"
    assert fields["w"].shape == w_shape, f"Expected w shape {w_shape}, got {fields['w'].shape}"
    assert fields["p"].shape == p_shape, f"Expected p shape {p_shape}, got {fields['p'].shape}"



