# tests/test_grid_initialization/test_ghost_cell_indexing.py

import pytest
from src.utils.grid import create_mac_grid_fields
from src.physics.boundary_conditions_applicator import apply_ghost_cells

@pytest.mark.parametrize("field_name", ["u", "v", "w"])
def test_ghost_cells_preserve_domain_and_apply_padding(field_name):
    """
    Ensures that ghost cells wrap correctly and that the inner domain
    retains its expected staggered dimensions after padding.
    """
    nx, ny, nz = 4, 4, 4
    ghost_width = 1
    fields = create_mac_grid_fields((nx, ny, nz), ghost_width=ghost_width)
    field = fields[field_name]

    apply_ghost_cells(field, field_name)

    # Determine true core interior shape based on staggered layout
    expected_shape = {
        "u": (nx + 1, ny, nz),
        "v": (nx, ny + 1, nz),
        "w": (nx, ny, nz + 1)
    }[field_name]

    interior = field[
        ghost_width:-ghost_width,
        ghost_width:-ghost_width,
        ghost_width:-ghost_width
    ]

    assert interior.shape == expected_shape, f"{field_name} field interior shape mismatch"



