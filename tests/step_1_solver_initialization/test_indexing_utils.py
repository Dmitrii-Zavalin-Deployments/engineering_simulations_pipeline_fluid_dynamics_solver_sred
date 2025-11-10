# tests/step_1_solver_initialization/test_indexing_utils.py
# ✅ Unit tests for step_1_solver_initialization/indexing_utils.py

import pytest
from step_1_solver_initialization.indexing_utils import (
    grid_to_flat,
    flat_to_grid,
    is_valid_grid_index,
    is_valid_flat_index
)

# 🔁 Round-trip conversion: flat → grid → flat
@pytest.mark.parametrize("shape", [
    (1, 1, 1),
    (2, 2, 2),
    (4, 3, 2),
    (5, 1, 1),
    (3, 3, 3),
])
def test_round_trip_flat_grid_flat(shape):
    nx, ny, nz = shape
    total_cells = nx * ny * nz
    for flat_index in range(total_cells):
        x, y, z = flat_to_grid(flat_index, shape)
        assert is_valid_grid_index(x, y, z, shape)
        reconverted = grid_to_flat(x, y, z, shape)
        assert reconverted == flat_index, f"Mismatch: {flat_index} → {[x,y,z]} → {reconverted}"

# 🔁 Round-trip conversion: grid → flat → grid
@pytest.mark.parametrize("shape", [
    (2, 2, 2),
    (4, 3, 2),
    (3, 1, 2),
])
def test_round_trip_grid_flat_grid(shape):
    nx, ny, nz = shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                flat = grid_to_flat(x, y, z, shape)
                x2, y2, z2 = flat_to_grid(flat, shape)
                assert [x, y, z] == [x2, y2, z2], f"Mismatch: {[x,y,z]} → {flat} → {[x2,y2,z2]}"

# 🧪 Known mappings
@pytest.mark.parametrize("x, y, z, shape, expected", [
    (0, 0, 0, (4, 3, 2), 0),
    (1, 0, 0, (4, 3, 2), 1),
    (0, 1, 0, (4, 3, 2), 4),
    (0, 0, 1, (4, 3, 2), 12),
    (3, 2, 1, (4, 3, 2), 23),
])
def test_grid_to_flat_known(x, y, z, shape, expected):
    assert grid_to_flat(x, y, z, shape) == expected

@pytest.mark.parametrize("flat_index, shape, expected", [
    (0, (4, 3, 2), [0, 0, 0]),
    (1, (4, 3, 2), [1, 0, 0]),
    (4, (4, 3, 2), [0, 1, 0]),
    (12, (4, 3, 2), [0, 0, 1]),
    (23, (4, 3, 2), [3, 2, 1]),
])
def test_flat_to_grid_known(flat_index, shape, expected):
    assert flat_to_grid(flat_index, shape) == expected

# ✅ Validity checks for grid index
@pytest.mark.parametrize("x, y, z, shape, expected", [
    (0, 0, 0, (4, 3, 2), True),
    (3, 2, 1, (4, 3, 2), True),
    (-1, 0, 0, (4, 3, 2), False),
    (4, 0, 0, (4, 3, 2), False),
    (0, 3, 0, (4, 3, 2), False),
    (0, 0, 2, (4, 3, 2), False),
])
def test_is_valid_grid_index(x, y, z, shape, expected):
    assert is_valid_grid_index(x, y, z, shape) == expected

# ✅ Validity checks for flat index
@pytest.mark.parametrize("flat_index, shape, expected", [
    (0, (4, 3, 2), True),
    (23, (4, 3, 2), True),
    (24, (4, 3, 2), False),
    (-1, (4, 3, 2), False),
])
def test_is_valid_flat_index(flat_index, shape, expected):
    assert is_valid_flat_index(flat_index, shape) == expected



