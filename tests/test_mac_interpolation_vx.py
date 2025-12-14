# tests/test_mac_interpolation_vx.py
# Unit tests for vx interpolation functions using the 3×3×3 mock cell_dict

import pytest
# ✅ Import from the package root, which now re-exports from vx.py
from src.step_2_time_stepping_loop.mac_interpolation import (
    vx_i_plus_half,
    vx_i_minus_half,
    vx_i_plus_three_half,
    vx_i_minus_three_half,
    vx_j_plus_one,
    vx_j_minus_one,
    vx_k_plus_one,
    vx_k_minus_one,
)
from tests.mocks.cell_dict_mock import cell_dict


def test_vx_i_plus_half_t0():
    result = vx_i_plus_half(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vx"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_i_plus_half_latest():
    result = vx_i_plus_half(cell_dict, 13)
    central = cell_dict[str(13)]["vx"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_i_minus_half_t0():
    result = vx_i_minus_half(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vx"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_i_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_i_minus_half_latest():
    result = vx_i_minus_half(cell_dict, 13)
    central = cell_dict[str(13)]["vx"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_i_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_i_plus_three_half_t0():
    result = vx_i_plus_three_half(cell_dict, 12, timestep=0)
    central_index = cell_dict[str(12)]["flat_index_i_plus_1"]
    central = cell_dict[str(central_index)]["vx"][0]
    neighbor_index = cell_dict[str(central_index)]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_i_plus_three_half_latest():
    result = vx_i_plus_three_half(cell_dict, 12)
    central_index = cell_dict[str(12)]["flat_index_i_plus_1"]
    central = cell_dict[str(central_index)]["vx"][1]
    neighbor_index = cell_dict[str(central_index)]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_i_minus_three_half_t0():
    result = vx_i_minus_three_half(cell_dict, 14, timestep=0)
    central_index = cell_dict[str(14)]["flat_index_i_minus_1"]
    central = cell_dict[str(central_index)]["vx"][0]
    neighbor_index = cell_dict[str(central_index)]["flat_index_i_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_i_minus_three_half_latest():
    result = vx_i_minus_three_half(cell_dict, 14)
    central_index = cell_dict[str(14)]["flat_index_i_minus_1"]
    central = cell_dict[str(central_index)]["vx"][1]
    neighbor_index = cell_dict[str(central_index)]["flat_index_i_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


# ---------------- New tests for cross-direction vx interpolations ----------------

def test_vx_j_plus_one_t0():
    result = vx_j_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vx"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_j_minus_one_latest():
    result = vx_j_minus_one(cell_dict, 13)
    central = cell_dict[str(13)]["vx"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_j_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_k_plus_one_t0():
    result = vx_k_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vx"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_k_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vx_k_minus_one_latest():
    result = vx_k_minus_one(cell_dict, 13)
    central = cell_dict[str(13)]["vx"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vx"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6



