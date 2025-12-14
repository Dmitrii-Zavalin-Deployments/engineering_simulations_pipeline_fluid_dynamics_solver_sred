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
    # Between central (13, vx=1.0) and right neighbor (14, vx=1.5)
    result = vx_i_plus_half(cell_dict, 13, timestep=0)
    assert abs(result - 0.5 * (1.0 + 1.5)) < 1e-6


def test_vx_i_plus_half_latest():
    # Defaults to timestep=1 (central vx=1.1, right vx=1.6)
    result = vx_i_plus_half(cell_dict, 13)
    assert abs(result - 0.5 * (1.1 + 1.6)) < 1e-6


def test_vx_i_minus_half_t0():
    # Between central (13, vx=1.0) and left neighbor (12, vx=0.5)
    result = vx_i_minus_half(cell_dict, 13, timestep=0)
    assert abs(result - 0.5 * (1.0 + 0.5)) < 1e-6


def test_vx_i_minus_half_latest():
    # Defaults to timestep=1 (central vx=1.1, left vx=0.6)
    result = vx_i_minus_half(cell_dict, 13)
    assert abs(result - 0.5 * (1.1 + 0.6)) < 1e-6


def test_vx_i_plus_three_half_t0():
    # Start from left neighbor (12) -> central (13) -> right (14)
    # Average vx from central (13) and right (14) at timestep=0
    result = vx_i_plus_three_half(cell_dict, 12, timestep=0)
    expected = 0.5 * (1.0 + 1.5)  # vx(13) + vx(14)
    assert abs(result - expected) < 1e-6


def test_vx_i_plus_three_half_latest():
    # Start from left neighbor (12) -> central (13) -> right (14)
    # Average vx from central (13) and right (14) at latest timestep=1
    result = vx_i_plus_three_half(cell_dict, 12)
    expected = 0.5 * (1.1 + 1.6)  # vx(13) + vx(14)
    assert abs(result - expected) < 1e-6


def test_vx_i_minus_three_half_t0():
    # Start from right neighbor (14) -> central (13) -> left (12)
    # Average vx from central (13) and left (12) at timestep=0
    result = vx_i_minus_three_half(cell_dict, 14, timestep=0)
    expected = 0.5 * (1.0 + 0.5)  # vx(13) + vx(12)
    assert abs(result - expected) < 1e-6


def test_vx_i_minus_three_half_latest():
    # Start from right neighbor (14) -> central (13) -> left (12)
    # Average vx from central (13) and left (12) at latest timestep=1
    result = vx_i_minus_three_half(cell_dict, 14)
    expected = 0.5 * (1.1 + 0.6)  # vx(13) + vx(12)
    assert abs(result - expected) < 1e-6


# ---------------- New tests for cross-direction vx interpolations ----------------

def test_vx_j_plus_one_t0():
    # Between central (13, vx=1.0) and j+1 neighbor (mocked as vx=1.2)
    result = vx_j_plus_one(cell_dict, 13, timestep=0)
    expected = 0.5 * (1.0 + 1.2)
    assert abs(result - expected) < 1e-6


def test_vx_j_minus_one_latest():
    # Between central (13, vx=1.1) and j-1 neighbor (mocked as vx=0.9)
    result = vx_j_minus_one(cell_dict, 13)
    expected = 0.5 * (1.1 + 0.9)
    assert abs(result - expected) < 1e-6


def test_vx_k_plus_one_t0():
    # Between central (13, vx=1.0) and k+1 neighbor (mocked as vx=1.4)
    result = vx_k_plus_one(cell_dict, 13, timestep=0)
    expected = 0.5 * (1.0 + 1.4)
    assert abs(result - expected) < 1e-6


def test_vx_k_minus_one_latest():
    # Between central (13, vx=1.1) and k-1 neighbor (mocked as vx=0.7)
    result = vx_k_minus_one(cell_dict, 13)
    expected = 0.5 * (1.1 + 0.7)
    assert abs(result - expected) < 1e-6



