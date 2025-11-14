# tests/test_indexing_utils.py
# ✅ Focused validation tests for x-major flattening logic in a 3×3×3 cube

import pytest
from step_1_solver_initialization.indexing_utils import (
    grid_to_flat,
    flat_to_grid,
    is_valid_grid_index,
    is_valid_flat_index,
)

# 🧩 Cube shape
CUBE_SHAPE = (3, 3, 3)

# --- Grid → Flat tests ---
@pytest.mark.parametrize("x, y, z, expected_flat", [
    # one non-zero coordinate
    (0, 0, 0, 0),
    (1, 0, 0, 1),
    (2, 0, 0, 2),
    (0, 1, 0, 3),
    (0, 2, 0, 6),
    (0, 0, 1, 9),
    (0, 0, 2, 18),
    # two non-zero coordinates, one zero
    (1, 1, 0, 4),
    (2, 1, 0, 5),
    (1, 0, 1, 10),
    (2, 0, 1, 11),
    (0, 1, 1, 12),
    (0, 2, 1, 15),
    # all non-zero coordinates
    (1, 1, 1, 13),
    (2, 2, 2, 26),
    # invalid coordinates (negative)
    (-1, 0, 0, None),
    (0, -1, 0, None),
    (0, 0, -1, None),
])
def test_grid_to_flat_cube(x, y, z, expected_flat):
    if is_valid_grid_index(x, y, z, CUBE_SHAPE):
        result = grid_to_flat(x, y, z, CUBE_SHAPE)
        assert result == expected_flat, f"Expected {expected_flat}, got {result}"
    else:
        assert expected_flat is None
        assert not is_valid_grid_index(x, y, z, CUBE_SHAPE)

# --- Flat → Grid tests ---
@pytest.mark.parametrize("flat_index, expected_grid", [
    # one non-zero coordinate
    (0, [0, 0, 0]),
    (1, [1, 0, 0]),
    (2, [2, 0, 0]),
    (3, [0, 1, 0]),
    (6, [0, 2, 0]),
    (9, [0, 0, 1]),
    (18, [0, 0, 2]),
    # two non-zero coordinates, one zero
    (4, [1, 1, 0]),
    (5, [2, 1, 0]),
    (10, [1, 0, 1]),
    (11, [2, 0, 1]),
    (12, [0, 1, 1]),
    (15, [0, 2, 1]),
    # all non-zero coordinates
    (13, [1, 1, 1]),
    (26, [2, 2, 2]),
    # invalid flat indices (negative only)
    (-1, None),
])
def test_flat_to_grid_cube(flat_index, expected_grid):
    if is_valid_flat_index(flat_index, CUBE_SHAPE):
        result = flat_to_grid(flat_index, CUBE_SHAPE)
        assert result == expected_grid, f"Expected {expected_grid}, got {result}"
    else:
        assert expected_grid is None
        assert not is_valid_flat_index(flat_index, CUBE_SHAPE)



