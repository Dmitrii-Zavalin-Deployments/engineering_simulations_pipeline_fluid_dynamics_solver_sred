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
    # Between central (13, vz=1.0) and above neighbor (22, vz=1.5)
    result = vz_k_plus_half(cell_dict, 13, timestep=0)
    assert abs(result - 0.5 * (1.0 + 1.5)) < 1e-6


def test_vz_k_plus_half_latest():
    # Defaults to timestep=1 (central vz=1.1, above vz=1.6)
    result = vz_k_plus_half(cell_dict, 13)
    assert abs(result - 0.5 * (1.1 + 1.6)) < 1e-6


def test_vz_k_minus_half_t0():
    # Between central (13, vz=1.0) and below neighbor (4, vz=0.5)
    result = vz_k_minus_half(cell_dict, 13, timestep=0)
    assert abs(result - 0.5 * (1.0 + 0.5)) < 1e-6


def test_vz_k_minus_half_latest():
    # Defaults to timestep=1 (central vz=1.1, below vz=0.6)
    result = vz_k_minus_half(cell_dict, 13)
    assert abs(result - 0.5 * (1.1 + 0.6)) < 1e-6


def test_vz_k_plus_three_half_t0():
    # Start from below neighbor (4) -> central (13) -> above (22)
    # Average vz from central (13) and above (22) at timestep=0
    result = vz_k_plus_three_half(cell_dict, 4, timestep=0)
    expected = 0.5 * (1.0 + 1.5)  # vz(13) + vz(22)
    assert abs(result - expected) < 1e-6


def test_vz_k_plus_three_half_latest():
    # Start from below neighbor (4) -> central (13) -> above (22)
    # Average vz from central (13) and above (22) at latest timestep=1
    result = vz_k_plus_three_half(cell_dict, 4)
    expected = 0.5 * (1.1 + 1.6)  # vz(13) + vz(22)
    assert abs(result - expected) < 1e-6


def test_vz_k_minus_three_half_t0():
    # Start from above neighbor (22) -> central (13) -> below (4)
    # Average vz from central (13) and below (4) at timestep=0
    result = vz_k_minus_three_half(cell_dict, 22, timestep=0)
    expected = 0.5 * (1.0 + 0.5)  # vz(13) + vz(4)
    assert abs(result - expected) < 1e-6


def test_vz_k_minus_three_half_latest():
    # Start from above neighbor (22) -> central (13) -> below (4)
    # Average vz from central (13) and below (4) at latest timestep=1
    result = vz_k_minus_three_half(cell_dict, 22)
    expected = 0.5 * (1.1 + 0.6)  # vz(13) + vz(4)
    assert abs(result - expected) < 1e-6


# ---------------- New tests for cross-direction vz interpolations ----------------

def test_vz_i_plus_one_t0():
    # Between central (13, vz=1.0) and i+1 neighbor (mocked as vz=1.3)
    result = vz_i_plus_one(cell_dict, 13, timestep=0)
    expected = 0.5 * (1.0 + 1.3)
    assert abs(result - expected) < 1e-6


def test_vz_i_minus_one_latest():
    # Between central (13, vz=1.1) and i-1 neighbor (mocked as vz=0.7)
    result = vz_i_minus_one(cell_dict, 13)
    expected = 0.5 * (1.1 + 0.7)
    assert abs(result - expected) < 1e-6


def test_vz_j_plus_one_t0():
    # Between central (13, vz=1.0) and j+1 neighbor (mocked as vz=1.2)
    result = vz_j_plus_one(cell_dict, 13, timestep=0)
    expected = 0.5 * (1.0 + 1.2)
    assert abs(result - expected) < 1e-6


def test_vz_j_minus_one_latest():
    # Between central (13, vz=1.1) and j-1 neighbor (mocked as vz=0.9)
    result = vz_j_minus_one(cell_dict, 13)
    expected = 0.5 * (1.1 + 0.9)
    assert abs(result - expected) < 1e-6



