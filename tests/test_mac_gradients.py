# tests/test_mac_gradients.py
# Unit tests for gradient and divergence operators using the 3×3×3 mock cell_dict

import pytest
from src.step_2_time_stepping_loop.mac_gradients import (
    grad_p_x,
    grad_p_y,
    grad_p_z,
    divergence,
)
from tests.mocks.cell_dict_mock import cell_dict


# ---------------- Pressure Gradient Tests ----------------

def test_grad_p_x_t0():
    dx = 1.0
    result = grad_p_x(cell_dict, 13, dx, timestep=0)
    expected = (cell_dict["14"]["time_history"]["0"]["pressure"]
                - cell_dict["13"]["time_history"]["0"]["pressure"]) / dx
    assert abs(result - expected) < 1e-6


def test_grad_p_x_latest():
    dx = 1.0
    result = grad_p_x(cell_dict, 13, dx)
    expected = (cell_dict["14"]["time_history"]["1"]["pressure"]
                - cell_dict["13"]["time_history"]["1"]["pressure"]) / dx
    assert abs(result - expected) < 1e-6


def test_grad_p_y_t0():
    dy = 1.0
    result = grad_p_y(cell_dict, 13, dy, timestep=0)
    expected = (cell_dict["16"]["time_history"]["0"]["pressure"]
                - cell_dict["13"]["time_history"]["0"]["pressure"]) / dy
    assert abs(result - expected) < 1e-6


def test_grad_p_z_t0():
    dz = 1.0
    result = grad_p_z(cell_dict, 13, dz, timestep=0)
    expected = (cell_dict["22"]["time_history"]["0"]["pressure"]
                - cell_dict["13"]["time_history"]["0"]["pressure"]) / dz
    assert abs(result - expected) < 1e-6


# ---------------- Boundary (Ghost-cell) Tests ----------------

def test_grad_p_x_boundary():
    dx = 1.0
    result = grad_p_x(cell_dict, 0, dx, timestep=0)
    # No neighbor → ghost cell pressure = current cell → gradient = 0
    assert abs(result) < 1e-12


def test_grad_p_y_boundary():
    dy = 1.0
    result = grad_p_y(cell_dict, 0, dy, timestep=0)
    # No neighbor → ghost cell pressure = current cell → gradient = 0
    assert abs(result) < 1e-12


def test_grad_p_z_boundary():
    dz = 1.0
    result = grad_p_z(cell_dict, 0, dz, timestep=0)
    # No neighbor → ghost cell pressure = current cell → gradient = 0
    assert abs(result) < 1e-12


# ---------------- Divergence Tests ----------------

def test_divergence_t0():
    dx = dy = dz = 1.0
    result = divergence(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    assert abs(result) < 10.0


def test_divergence_latest():
    dx = dy = dz = 1.0
    result = divergence(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert abs(result) < 10.0



