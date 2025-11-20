# tests/test_mac_diffusion.py
# Unit tests for diffusion (Laplacian) operators using the 3×3×3 mock cell_dict

import pytest
from src.step_2_time_stepping_loop.mac_diffusion import (
    laplacian_vx,
    laplacian_vy,
    laplacian_vz,
    laplacian_velocity,
)
from tests.mocks.cell_dict_mock import cell_dict


# ---------------- Laplacian of Velocity Components ----------------

def test_laplacian_vx_t0():
    dx = 1.0
    result = laplacian_vx(cell_dict, 13, dx, timestep=0)
    assert isinstance(result, float)


def test_laplacian_vx_latest():
    dx = 1.0
    result = laplacian_vx(cell_dict, 13, dx)
    assert isinstance(result, float)


def test_laplacian_vy_t0():
    dy = 1.0
    result = laplacian_vy(cell_dict, 13, dy, timestep=0)
    assert isinstance(result, float)


def test_laplacian_vy_latest():
    dy = 1.0
    result = laplacian_vy(cell_dict, 13, dy)
    assert isinstance(result, float)


def test_laplacian_vz_t0():
    dz = 1.0
    result = laplacian_vz(cell_dict, 13, dz, timestep=0)
    assert isinstance(result, float)


def test_laplacian_vz_latest():
    dz = 1.0
    result = laplacian_vz(cell_dict, 13, dz)
    assert isinstance(result, float)


# ---------------- Boundary (Ghost-cell) Tests ----------------

def test_laplacian_vx_boundary():
    dx = 1.0
    result = laplacian_vx(cell_dict, 0, dx, timestep=0)
    # With no neighbors, Laplacian should degrade gracefully (finite value)
    assert isinstance(result, float)


def test_laplacian_vy_boundary():
    dy = 1.0
    result = laplacian_vy(cell_dict, 0, dy, timestep=0)
    assert isinstance(result, float)


def test_laplacian_vz_boundary():
    dz = 1.0
    result = laplacian_vz(cell_dict, 0, dz, timestep=0)
    assert isinstance(result, float)


# ---------------- Vector Laplacian ----------------

def test_laplacian_velocity_t0():
    dx = dy = dz = 1.0
    result = laplacian_velocity(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"vx", "vy", "vz"}


def test_laplacian_velocity_latest():
    dx = dy = dz = 1.0
    result = laplacian_velocity(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"vx", "vy", "vz"}



