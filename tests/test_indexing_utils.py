# tests/test_indexing_utils.py
# ✅ Focused validation tests for x-major flattening logic in a 3×3×3 cube + edge cases

import pytest
from step_1_solver_initialization import indexing_utils
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
    # invalid coordinates (beyond upper bound)
    (3, 0, 0, None),
    (0, 3, 0, None),
    (0, 0, 3, None),
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
    # invalid flat indices
    (-1, None),
    (27, None),  # beyond upper bound
    (100, None), # far beyond
])
def test_flat_to_grid_cube(flat_index, expected_grid):
    if is_valid_flat_index(flat_index, CUBE_SHAPE):
        result = flat_to_grid(flat_index, CUBE_SHAPE)
        assert result == expected_grid, f"Expected {expected_grid}, got {result}"
    else:
        assert expected_grid is None
        assert not is_valid_flat_index(flat_index, CUBE_SHAPE)

# --- Round-trip consistency ---
def test_round_trip_cube():
    nx, ny, nz = CUBE_SHAPE
    total = nx * ny * nz
    for flat_index in range(total):
        coords = flat_to_grid(flat_index, CUBE_SHAPE)
        reconverted = grid_to_flat(*coords, CUBE_SHAPE)
        assert reconverted == flat_index

# --- Boundary values ---
def test_boundary_values_cube():
    nx, ny, nz = CUBE_SHAPE
    last_index = nx * ny * nz - 1
    assert grid_to_flat(nx-1, ny-1, nz-1, CUBE_SHAPE) == last_index
    assert flat_to_grid(last_index, CUBE_SHAPE) == [nx-1, ny-1, nz-1]

# --- Debug flag coverage ---
def test_debug_flag(monkeypatch, capsys):
    indexing_utils.debug = True
    result = grid_to_flat(1, 1, 1, CUBE_SHAPE)
    assert result == 13
    result2 = flat_to_grid(13, CUBE_SHAPE)
    assert result2 == [1, 1, 1]
    result3 = is_valid_grid_index(1, 1, 1, CUBE_SHAPE)
    assert result3 is True
    result4 = is_valid_flat_index(13, CUBE_SHAPE)
    assert result4 is True
    captured = capsys.readouterr()
    assert "📐 grid_to_flat" in captured.out
    assert "📐 flat_to_grid" in captured.out
    assert "🔍 is_valid_grid_index" in captured.out
    assert "🔍 is_valid_flat_index" in captured.out
    indexing_utils.debug = False



