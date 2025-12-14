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
    # Between central (13, vy=1.0) and up neighbor (16, vy=1.5)
    result = vy_j_plus_half(cell_dict, 13, timestep=0)
    assert abs(result - 0.5 * (1.0 + 1.5)) < 1e-6


def test_vy_j_plus_half_latest():
    # Defaults to timestep=1 (central vy=1.1, up vy=1.6)
    result = vy_j_plus_half(cell_dict, 13)
    assert abs(result - 0.5 * (1.1 + 1.6)) < 1e-6


def test_vy_j_minus_half_t0():
    # Between central (13, vy=1.0) and down neighbor (10, vy=0.5)
    result = vy_j_minus_half(cell_dict, 13, timestep=0)
    assert abs(result - 0.5 * (1.0 + 0.5)) < 1e-6


def test_vy_j_minus_half_latest():
    # Defaults to timestep=1 (central vy=1.1, down vy=0.6)
    result = vy_j_minus_half(cell_dict, 13)
    assert abs(result - 0.5 * (1.1 + 0.6)) < 1e-6


def test_vy_j_plus_three_half_t0():
    # Start from down neighbor (10) -> central (13) -> up (16)
    # Average vy from central (13) and up (16) at timestep=0
    result = vy_j_plus_three_half(cell_dict, 10, timestep=0)
    expected = 0.5 * (1.0 + 1.5)  # vy(13) + vy(16)
    assert abs(result - expected) < 1e-6


def test_vy_j_plus_three_half_latest():
    # Start from down neighbor (10) -> central (13) -> up (16)
    # Average vy from central (13) and up (16) at latest timestep=1
    result = vy_j_plus_three_half(cell_dict, 10)
    expected = 0.5 * (1.1 + 1.6)  # vy(13) + vy(16)
    assert abs(result - expected) < 1e-6


def test_vy_j_minus_three_half_t0():
    # Start from up neighbor (16) -> central (13) -> down (10)
    # Average vy from central (13) and down (10) at timestep=0
    result = vy_j_minus_three_half(cell_dict, 16, timestep=0)
    expected = 0.5 * (1.0 + 0.5)  # vy(13) + vy(10)
    assert abs(result - expected) < 1e-6


def test_vy_j_minus_three_half_latest():
    # Start from up neighbor (16) -> central (13) -> down (10)
    # Average vy from central (13) and down (10) at latest timestep=1
    result = vy_j_minus_three_half(cell_dict, 16)
    expected = 0.5 * (1.1 + 0.6)  # vy(13) + vy(10)
    assert abs(result - expected) < 1e-6


# ---------------- New tests for cross-direction vy interpolations ----------------

def test_vy_i_plus_one_t0():
    # Between central (13, vy=1.0) and i+1 neighbor (mocked as vy=1.3)
    result = vy_i_plus_one(cell_dict, 13, timestep=0)
    expected = 0.5 * (1.0 + 1.3)
    assert abs(result - expected) < 1e-6


def test_vy_i_minus_one_latest():
    # Between central (13, vy=1.1) and i-1 neighbor (mocked as vy=0.7)
    result = vy_i_minus_one(cell_dict, 13)
    expected = 0.5 * (1.1 + 0.7)
    assert abs(result - expected) < 1e-6


def test_vy_k_plus_one_t0():
    # Between central (13, vy=1.0) and k+1 neighbor (mocked as vy=1.4)
    result = vy_k_plus_one(cell_dict, 13, timestep=0)
    expected = 0.5 * (1.0 + 1.4)
    assert abs(result - expected) < 1e-6


def test_vy_k_minus_one_latest():
    # Between central (13, vy=1.1) and k-1 neighbor (mocked as vy=0.8)
    result = vy_k_minus_one(cell_dict, 13)
    expected = 0.5 * (1.1 + 0.8)
    assert abs(result - expected) < 1e-6



