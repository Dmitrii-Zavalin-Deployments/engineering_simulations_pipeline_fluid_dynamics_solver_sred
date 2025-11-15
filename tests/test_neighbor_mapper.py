# tests/test_neighbor_mapper.py
# ✅ Single tests for neighbors in a 3×3×3 cube

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


def test_neighbors_of_index_thirteen_cube():
    shape = (3, 3, 3)
    flat_index = 13  # corresponds to (1,1,1)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_plus_1": 14,
        "flat_index_i_minus_1": 12,
        "flat_index_j_plus_1": 16,
        "flat_index_j_minus_1": 10,
        "flat_index_k_plus_1": 22,
        "flat_index_k_minus_1": 4,
    }

    assert neighbors == expected


def test_neighbors_of_index_twenty_six_cube():
    shape = (3, 3, 3)
    flat_index = 26  # corresponds to (2,2,2)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_plus_1": None,
        "flat_index_i_minus_1": 25,
        "flat_index_j_plus_1": None,
        "flat_index_j_minus_1": 23,
        "flat_index_k_plus_1": None,
        "flat_index_k_minus_1": 17,
    }

    assert neighbors == expected


def test_neighbors_edge_x_axis():
    shape = (3, 3, 3)
    flat_index = 14  # corresponds to (2,1,1)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_plus_1": None,
        "flat_index_i_minus_1": 13,
        "flat_index_j_plus_1": 17,
        "flat_index_j_minus_1": 11,
        "flat_index_k_plus_1": 23,
        "flat_index_k_minus_1": 5,
    }

    assert neighbors == expected


def test_neighbors_edge_y_axis():
    shape = (3, 3, 3)
    flat_index = 16  # corresponds to (1,2,1)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_plus_1": 17,
        "flat_index_i_minus_1": 15,
        "flat_index_j_plus_1": None,
        "flat_index_j_minus_1": 13,
        "flat_index_k_plus_1": 25,
        "flat_index_k_minus_1": 7,
    }

    assert neighbors == expected


def test_neighbors_edge_z_axis():
    shape = (3, 3, 3)
    flat_index = 22  # corresponds to (1,1,2)
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_plus_1": 23,
        "flat_index_i_minus_1": 21,
        "flat_index_j_plus_1": 25,
        "flat_index_j_minus_1": 19,
        "flat_index_k_plus_1": None,
        "flat_index_k_minus_1": 13,
    }

    assert neighbors == expected


def test_single_cell_domain():
    shape = (1, 1, 1)
    flat_index = 0
    neighbors = get_stencil_neighbors(flat_index, shape)

    expected = {
        "flat_index_i_plus_1": None,
        "flat_index_i_minus_1": None,
        "flat_index_j_plus_1": None,
        "flat_index_j_minus_1": None,
        "flat_index_k_plus_1": None,
        "flat_index_k_minus_1": None,
    }

    assert neighbors == expected


def test_round_trip_consistency_cube():
    shape = (3, 3, 3)
    for flat_index in range(shape[0] * shape[1] * shape[2]):
        neighbors = get_stencil_neighbors(flat_index, shape)
        # All non-None neighbors must be within valid range
        for neighbor in neighbors.values():
            if neighbor is not None:
                assert 0 <= neighbor < shape[0] * shape[1] * shape[2]



