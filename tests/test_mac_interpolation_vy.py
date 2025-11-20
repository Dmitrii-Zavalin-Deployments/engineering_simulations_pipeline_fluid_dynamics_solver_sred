# tests/test_mac_interpolation_vy.py
# Unit tests for vy interpolation functions using the 3×3×3 mock cell_dict

import pytest
from src.step_2_time_stepping_loop.mac_interpolation import (
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
    vy_j_minus_three_half,   # <-- added correct import
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



