# tests/test_mac_advection.py
# Unit tests for advection (nonlinear convective terms) using the 3×3×3 mock cell_dict

import pytest
from src.step_2_time_stepping_loop.mac_advection import (
    adv_vx,
    adv_vy,
    adv_vz,
)
from tests.mocks.cell_dict_mock import cell_dict


# ---------------- Central cell tests ----------------

def test_adv_vx_central_t0():
    dx = dy = dz = 1.0
    result = adv_vx(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert abs(result) < 50.0  # sanity bound


def test_adv_vx_central_latest():
    dx = dy = dz = 1.0
    result = adv_vx(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert abs(result) < 50.0


def test_adv_vy_central_t0():
    dx = dy = dz = 1.0
    result = adv_vy(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert abs(result) < 50.0


def test_adv_vy_central_latest():
    dx = dy = dz = 1.0
    result = adv_vy(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert abs(result) < 50.0


def test_adv_vz_central_t0():
    dx = dy = dz = 1.0
    result = adv_vz(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert abs(result) < 50.0


def test_adv_vz_central_latest():
    dx = dy = dz = 1.0
    result = adv_vz(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert abs(result) < 50.0


# ---------------- Boundary (ghost-cell) tests ----------------

def test_adv_vx_boundary():
    dx = dy = dz = 1.0
    result = adv_vx(cell_dict, 0, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    # ghost cell should degrade gracefully
    assert abs(result) < 1e-6 or abs(result) < 50.0


def test_adv_vy_boundary():
    dx = dy = dz = 1.0
    result = adv_vy(cell_dict, 0, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert abs(result) < 1e-6 or abs(result) < 50.0


def test_adv_vz_boundary():
    dx = dy = dz = 1.0
    result = adv_vz(cell_dict, 0, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert abs(result) < 1e-6 or abs(result) < 50.0


# ---------------- Structure and stability checks ----------------

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



