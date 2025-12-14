# tests/test_mac_interpolation_vz.py
# Unit tests for vz interpolation functions using the 3×3×3 mock cell_dict

import pytest
# ✅ Import from the package root, which now re-exports from vz.py
from src.step_2_time_stepping_loop.mac_interpolation import (
    vz_k_plus_half,
    vz_k_minus_half,
    vz_k_plus_three_half,
    vz_k_minus_three_half,
    vz_i_plus_one,
    vz_i_minus_one,
    vz_j_plus_one,
    vz_j_minus_one,
)
from tests.mocks.cell_dict_mock import cell_dict


def test_vz_k_plus_half_t0():
    result = vz_k_plus_half(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vz"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_k_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_k_plus_half_latest():
    result = vz_k_plus_half(cell_dict, 13)
    central = cell_dict[str(13)]["vz"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_k_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_k_minus_half_t0():
    result = vz_k_minus_half(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vz"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_k_minus_half_latest():
    result = vz_k_minus_half(cell_dict, 13)
    central = cell_dict[str(13)]["vz"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_k_plus_three_half_t0():
    result = vz_k_plus_three_half(cell_dict, 4, timestep=0)
    central_index = cell_dict[str(4)]["flat_index_k_plus_1"]
    central = cell_dict[str(central_index)]["vz"][0]
    neighbor_index = cell_dict[str(central_index)]["flat_index_k_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_k_plus_three_half_latest():
    result = vz_k_plus_three_half(cell_dict, 4)
    central_index = cell_dict[str(4)]["flat_index_k_plus_1"]
    central = cell_dict[str(central_index)]["vz"][1]
    neighbor_index = cell_dict[str(central_index)]["flat_index_k_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_k_minus_three_half_t0():
    result = vz_k_minus_three_half(cell_dict, 22, timestep=0)
    central_index = cell_dict[str(22)]["flat_index_k_minus_1"]
    central = cell_dict[str(central_index)]["vz"][0]
    neighbor_index = cell_dict[str(central_index)]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_k_minus_three_half_latest():
    result = vz_k_minus_three_half(cell_dict, 22)
    central_index = cell_dict[str(22)]["flat_index_k_minus_1"]
    central = cell_dict[str(central_index)]["vz"][1]
    neighbor_index = cell_dict[str(central_index)]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


# ---------------- Refactored cross-direction vz interpolations ----------------

def test_vz_i_plus_one_t0():
    result = vz_i_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vz"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_i_minus_one_latest():
    result = vz_i_minus_one(cell_dict, 13)
    central = cell_dict[str(13)]["vz"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_i_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_j_plus_one_t0():
    result = vz_j_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vz"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vz_j_minus_one_latest():
    result = vz_j_minus_one(cell_dict, 13)
    central = cell_dict[str(13)]["vz"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_j_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vz"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6



