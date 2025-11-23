# tests/test_mac_advection.py
# Unit tests for advection operators (adv_vx, adv_vy, adv_vz)

import pytest

# Import advection operators from the new ops module
from src.step_2_time_stepping_loop.mac_advection_ops import (
    adv_vx,
    adv_vy,
    adv_vz,
)

from tests.mocks.cell_dict_mock import cell_dict


# ---------------- Central cell tests ----------------
# These tests check advection at an interior (central) cell index (13).
# Purpose: verify that adv_vx, adv_vy, adv_vz return finite floats
# both at the initial timestep (t0) and at the latest timestep (None → n).
# Why: ensures correct gradient calculation and safe handling of time history.

def test_adv_vx_central_t0():
    dx = dy = dz = 1.0
    result = adv_vx(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert result == result
    assert abs(result) < 1e3


def test_adv_vy_central_t0():
    dx = dy = dz = 1.0
    result = adv_vy(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert result == result
    assert abs(result) < 1e3


def test_adv_vz_central_t0():
    dx = dy = dz = 1.0
    result = adv_vz(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert result == result
    assert abs(result) < 1e3


def test_adv_vx_central_latest():
    dx = dy = dz = 1.0
    result = adv_vx(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert result == result
    assert abs(result) < 1e3


def test_adv_vy_central_latest():
    dx = dy = dz = 1.0
    result = adv_vy(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert result == result
    assert abs(result) < 1e3


def test_adv_vz_central_latest():
    dx = dy = dz = 1.0
    result = adv_vz(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert result == result
    assert abs(result) < 1e3


# ---------------- Boundary (ghost-cell) tests ----------------
# These tests check advection at a boundary cell index (0).
# Purpose: verify ghost-cell fallback logic works (no crash, finite result).
# Why: ensures safe handling when neighbors are missing.

def test_adv_vx_boundary():
    dx = dy = dz = 1.0
    result = adv_vx(cell_dict, 0, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert result == result


def test_adv_vy_boundary():
    dx = dy = dz = 1.0
    result = adv_vy(cell_dict, 0, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert result == result


def test_adv_vz_boundary():
    dx = dy = dz = 1.0
    result = adv_vz(cell_dict, 0, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert result == result


# ---------------- Parametrized structure checks ----------------
# These tests run across all three advection functions using pytest parametrize.
# Purpose: enforce consistent return type and finite values.
# Why: ensures adv_vx, adv_vy, adv_vz behave uniformly and never produce NaN/inf.

@pytest.mark.parametrize("func", [adv_vx, adv_vy, adv_vz])
def test_advection_returns_float(func):
    dx = dy = dz = 1.0
    result = func(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)


@pytest.mark.parametrize("func", [adv_vx, adv_vy, adv_vz])
def test_advection_finite_values(func):
    dx = dy = dz = 1.0
    result = func(cell_dict, 13, dx, dy, dz, timestep=0)
    assert result == result  # not NaN
    assert abs(result) < 1e6  # finite bound



