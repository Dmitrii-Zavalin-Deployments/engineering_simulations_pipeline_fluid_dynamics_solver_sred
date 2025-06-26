# tests/test_grid_initialization/test_ghost_cell_indexing.py

import numpy as np
import pytest

# 🔧 Adjust these import paths based on your actual module layout
from src.utils import grid
from src.physics.boundary_conditions_applicator import apply_ghost_cells
@pytest.mark.parametrize("field_name", ["u", "v", "w"])
def test_ghost_cells_preserve_domain_and_apply_padding(field_name):
    """
    Validates that ghost cells are correctly added around the velocity field.
    The real solver's functions are used to allocate and apply ghost cells.
    """
    nx, ny, nz = 4, 4, 4
    ghost_width = 1

    # Step 1: Generate core MAC grid velocity fields
    fields = create_mac_grid_fields((nx, ny, nz), ghost_width=ghost_width)
    field = fields[field_name]  # e.g., fields["u"]

    # Step 2: Apply boundary padding using solver logic
    apply_ghost_cells(field, field_name)

    # Step 3: Check that inner domain is preserved
    interior = field[ghost_width:-ghost_width, ghost_width:-ghost_width, ghost_width:-ghost_width]
    assert interior.shape == (nx, ny, nz), "Interior domain has incorrect shape after padding."
    assert not np.any(np.isnan(interior)), "Interior domain contains NaNs."

    # Step 4: Check that ghost cells are non-zero (or marked)
    outer_shell = field.copy()
    outer_shell[ghost_width:-ghost_width, ghost_width:-ghost_width, ghost_width:-ghost_width] = 0
    assert np.any(outer_shell != 0), f"Ghost cells for {field_name} field may not have been set properly."



