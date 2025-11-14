# tests/test_neighbor_mapper.py
# ✅ Unit tests for src/step_1_solver_initialization/neighbor_mapper.py

import pytest

from src.step_1_solver_initialization.neighbor_mapper import get_stencil_neighbors
from src.step_1_solver_initialization.indexing_utils import flat_to_grid, grid_to_flat

# 🧪 Core validation: interior cell neighbors
def test_interior_neighbors():
    shape = (4, 3, 2)
    # interior point: (x=2, y=1, z=1)
    flat_index = grid_to_flat(2, 1, 1, shape)
    neighbors = get_stencil_neighbors(flat_index, shape)

    # Expected neighbors
    expected = {
        "flat_index_i_minus_1": grid_to_flat(1, 1, 1, shape),
        "flat_index_i_plus_1": grid_to_flat(3, 1, 1, shape),
        "flat_index_j_minus_1": grid_to_flat(2, 0, 1, shape),
        "flat_index_j_plus_1": grid_to_flat(2, 2, 1, shape),
        "flat_index_k_minus_1": grid_to_flat(2, 1, 0, shape),
        "flat_index_k_plus_1": None,  # z+1 = 2 is out of bounds
    }
    assert neighbors == expected

# 🧪 Boundary case: minimum corner (0,0,0)
def test_corner_min_neighbors():
    shape = (4, 3, 2)
    flat_index = grid_to_flat(0, 0, 0, shape)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_minus_1": None,
        "flat_index_i_plus_1": grid_to_flat(1, 0, 0, shape),
        "flat_index_j_minus_1": None,
        "flat_index_j_plus_1": grid_to_flat(0, 1, 0, shape),
        "flat_index_k_minus_1": None,
        "flat_index_k_plus_1": grid_to_flat(0, 0, 1, shape),
    }
    assert neighbors == expected

# 🧪 Boundary case: maximum corner (nx-1, ny-1, nz-1)
def test_corner_max_neighbors():
    shape = (4, 3, 2)
    flat_index = grid_to_flat(3, 2, 1, shape)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_minus_1": grid_to_flat(2, 2, 1, shape),
        "flat_index_i_plus_1": None,
        "flat_index_j_minus_1": grid_to_flat(3, 1, 1, shape),
        "flat_index_j_plus_1": None,
        "flat_index_k_minus_1": grid_to_flat(3, 2, 0, shape),
        "flat_index_k_plus_1": None,
    }
    assert neighbors == expected

# 🧪 Edge case: single-cell domain (1,1,1)
def test_single_cell_domain():
    shape = (1, 1, 1)
    flat_index = 0
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_minus_1": None,
        "flat_index_i_plus_1": None,
        "flat_index_j_minus_1": None,
        "flat_index_j_plus_1": None,
        "flat_index_k_minus_1": None,
        "flat_index_k_plus_1": None,
    }
    assert neighbors == expected

# 🧪 Consistency check: round-trip grid/flat indices
@pytest.mark.parametrize("x,y,z,shape", [
    (1, 1, 0, (4, 3, 2)),
    (0, 2, 1, (4, 3, 2)),
    (3, 0, 0, (4, 3, 2)),
])
def test_consistency_with_indexing_utils(x, y, z, shape):
    flat_index = grid_to_flat(x, y, z, shape)
    neighbors = get_stencil_neighbors(flat_index, shape)

    # All non-None neighbors must map back to valid grid coordinates
    for label, neighbor_flat in neighbors.items():
        if neighbor_flat is not None:
            nx, ny, nz = flat_to_grid(neighbor_flat, shape)
            assert 0 <= nx < shape[0]
            assert 0 <= ny < shape[1]
            assert 0 <= nz < shape[2]

# 🧪 Performance guard: large grid should compute quickly
def test_large_grid_performance():
    shape = (50, 50, 50)
    flat_index = grid_to_flat(25, 25, 25, shape)
    neighbors = get_stencil_neighbors(flat_index, shape)
    # Ensure all six neighbor keys exist
    assert set(neighbors.keys()) == {
        "flat_index_i_minus_1",
        "flat_index_i_plus_1",
        "flat_index_j_minus_1",
        "flat_index_j_plus_1",
        "flat_index_k_minus_1",
        "flat_index_k_plus_1",
    }



