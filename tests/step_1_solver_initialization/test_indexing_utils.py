# tests/step_1_solver_initialization/test_indexing_utils.py
# ✅ Validation tests for x-major flattening logic in indexing_utils.py

import pytest
from step_1_solver_initialization.indexing_utils import (
    grid_to_flat,
    flat_to_grid,
    is_valid_grid_index,
    is_valid_flat_index
)

def test_reference_example_flat_to_grid():
    # 📐 Reference: nx=4, ny=3, nz=2, flat_index=23 → (x=3, y=2, z=1)
    shape = (4, 3, 2)
    flat_index = 23
    expected_grid = [3, 2, 1]
    result = flat_to_grid(flat_index, shape)
    assert result == expected_grid, f"Expected {expected_grid}, got {result}"

def test_reference_example_grid_to_flat():
    # 📐 Reference: (x=3, y=2, z=1) → flat_index=23
    shape = (4, 3, 2)
    x, y, z = 3, 2, 1
    expected_flat = 23
    result = grid_to_flat(x, y, z, shape)
    assert result == expected_flat, f"Expected {expected_flat}, got {result}"

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
        reconverted = grid_to_flat(x, y, z, shape)
        assert reconverted == flat_index, f"Mismatch: {flat_index} → {[x,y,z]} → {reconverted}"

@pytest.mark.parametrize("flat_index, shape, expected", [
    (0, (4, 3, 2), [0, 0, 0]),
    (1, (4, 3, 2), [1, 0, 0]),
    (4, (4, 3, 2), [0, 1, 0]),
    (12, (4, 3, 2), [0, 0, 1]),
    (23, (4, 3, 2), [3, 2, 1]),
])
def test_flat_to_grid_known(flat_index, shape, expected):
    result = flat_to_grid(flat_index, shape)
    assert result == expected, f"Expected {expected}, got {result}"

@pytest.mark.parametrize("x, y, z, shape, expected", [
    (0, 0, 0, (4, 3, 2), 0),
    (1, 0, 0, (4, 3, 2), 1),
    (0, 1, 0, (4, 3, 2), 4),
    (0, 0, 1, (4, 3, 2), 12),
    (3, 2, 1, (4, 3, 2), 23),
])
def test_grid_to_flat_known(x, y, z, shape, expected):
    result = grid_to_flat(x, y, z, shape)
    assert result == expected, f"Expected {expected}, got {result}"

@pytest.mark.parametrize("x, y, z, shape, expected", [
    (0, 0, 0, (4, 3, 2), True),
    (3, 2, 1, (4, 3, 2), True),
    (-1, 0, 0, (4, 3, 2), False),
    (4, 0, 0, (4, 3, 2), False),
    (0, 3, 0, (4, 3, 2), False),
    (0, 0, 2, (4, 3, 2), False),
])
def test_is_valid_grid_index(x, y, z, shape, expected):
    result = is_valid_grid_index(x, y, z, shape)
    assert result == expected, f"Expected {expected}, got {result}"

@pytest.mark.parametrize("flat_index, shape, expected", [
    (0, (4, 3, 2), True),
    (23, (4, 3, 2), True),
    (24, (4, 3, 2), False),
    (-1, (4, 3, 2), False),
])
def test_is_valid_flat_index(flat_index, shape, expected):
    result = is_valid_flat_index(flat_index, shape)
    assert result == expected, f"Expected {expected}, got {result}"



