# tests/test_neighbor_mapper.py
# ✅ Single test for neighbors of cell with flat index 0 in a 3×3×3 cube

from step_1_solver_initialization.neighbor_mapper import get_stencil_neighbors

def test_neighbors_of_index_zero_cube():
    shape = (3, 3, 3)
    flat_index = 0  # corresponds to (0,0,0)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_plus_1": 1,
        "flat_index_i_minus_1": None,
        "flat_index_j_plus_1": 3,
        "flat_index_j_minus_1": None,
        "flat_index_k_plus_1": 9,
        "flat_index_k_minus_1": None,
    }

    assert neighbors == expected



