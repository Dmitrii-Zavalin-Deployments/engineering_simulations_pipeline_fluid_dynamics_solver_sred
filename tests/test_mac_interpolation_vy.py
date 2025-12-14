# tests/test_mac_interpolation_vy.py
# Unit tests for vy interpolation functions using the 3×3×3 mock cell_dict

import pytest
# ✅ Import from the package root, which now re-exports from vy.py
from src.step_2_time_stepping_loop.mac_interpolation import (
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
    vy_j_minus_three_half,
    vy_i_plus_one,
    vy_i_minus_one,
    vy_k_plus_one,
    vy_k_minus_one,
)
from tests.mocks.cell_dict_mock import cell_dict


def test_vy_j_plus_half_t0():
    result = vy_j_plus_half(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vy"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_j_plus_half_latest():
    result = vy_j_plus_half(cell_dict, 13)
    central = cell_dict[str(13)]["vy"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_j_minus_half_t0():
    result = vy_j_minus_half(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vy"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_j_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_j_minus_half_latest():
    result = vy_j_minus_half(cell_dict, 13)
    central = cell_dict[str(13)]["vy"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_j_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_j_plus_three_half_t0():
    result = vy_j_plus_three_half(cell_dict, 10, timestep=0)
    central_index = cell_dict[str(10)]["flat_index_j_plus_1"]
    central = cell_dict[str(central_index)]["vy"][0]
    neighbor_index = cell_dict[str(central_index)]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_j_plus_three_half_latest():
    result = vy_j_plus_three_half(cell_dict, 10)
    central_index = cell_dict[str(10)]["flat_index_j_plus_1"]
    central = cell_dict[str(central_index)]["vy"][1]
    neighbor_index = cell_dict[str(central_index)]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_j_minus_three_half_t0():
    result = vy_j_minus_three_half(cell_dict, 16, timestep=0)
    central_index = cell_dict[str(16)]["flat_index_j_minus_1"]
    central = cell_dict[str(central_index)]["vy"][0]
    neighbor_index = cell_dict[str(central_index)]["flat_index_j_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_j_minus_three_half_latest():
    result = vy_j_minus_three_half(cell_dict, 16)
    central_index = cell_dict[str(16)]["flat_index_j_minus_1"]
    central = cell_dict[str(central_index)]["vy"][1]
    neighbor_index = cell_dict[str(central_index)]["flat_index_j_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


# ---------------- Refactored cross-direction vy interpolations ----------------

def test_vy_i_plus_one_t0():
    result = vy_i_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vy"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_i_minus_one_latest():
    result = vy_i_minus_one(cell_dict, 13)
    central = cell_dict[str(13)]["vy"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_i_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_k_plus_one_t0():
    result = vy_k_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict[str(13)]["vy"][0]
    neighbor_index = cell_dict[str(13)]["flat_index_k_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][0]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6


def test_vy_k_minus_one_latest():
    result = vy_k_minus_one(cell_dict, 13)
    central = cell_dict[str(13)]["vy"][1]
    neighbor_index = cell_dict[str(13)]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["vy"][1]
    expected = 0.5 * (central + neighbor)
    assert abs(result - expected) < 1e-6



