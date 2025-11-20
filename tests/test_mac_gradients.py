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
    # Central (13) vs right neighbor (14) at timestep=0
    dx = 1.0
    result = grad_p_x(cell_dict, 13, dx, timestep=0)
    expected = (cell_dict["14"]["time_history"]["0"]["pressure"]
                - cell_dict["13"]["time_history"]["0"]["pressure"]) / dx
    assert abs(result - expected) < 1e-6


def test_grad_p_x_latest():
    # Defaults to timestep=1
    dx = 1.0
    result = grad_p_x(cell_dict, 13, dx)
    expected = (cell_dict["14"]["time_history"]["1"]["pressure"]
                - cell_dict["13"]["time_history"]["1"]["pressure"]) / dx
    assert abs(result - expected) < 1e-6


def test_grad_p_y_t0():
    # Central (13) vs up neighbor (16) at timestep=0
    dy = 1.0
    result = grad_p_y(cell_dict, 13, dy, timestep=0)
    expected = (cell_dict["16"]["time_history"]["0"]["pressure"]
                - cell_dict["13"]["time_history"]["0"]["pressure"]) / dy
    assert abs(result - expected) < 1e-6


def test_grad_p_z_t0():
    # Central (13) vs above neighbor (22) at timestep=0
    dz = 1.0
    result = grad_p_z(cell_dict, 13, dz, timestep=0)
    expected = (cell_dict["22"]["time_history"]["0"]["pressure"]
                - cell_dict["13"]["time_history"]["0"]["pressure"]) / dz
    assert abs(result - expected) < 1e-6


def test_grad_p_x_boundary():
    # Ghost-cell case: leftmost cell with no i+1 neighbor
    dx = 1.0
    # Pick a cell with flat_index_i_plus_1 = None
    boundary_cell = "0"
    result = grad_p_x(cell_dict, int(boundary_cell), dx, timestep=0)
    # Should be zero since p_ip1 == p_i
    assert abs(result) < 1e-12


# ---------------- Divergence Tests ----------------

def test_divergence_t0():
    dx = dy = dz = 1.0
    result = divergence(cell_dict, 13, dx, dy, dz, timestep=0)
    assert isinstance(result, float)
    # Just check it's finite and consistent
    assert abs(result) < 10.0


def test_divergence_latest():
    dx = dy = dz = 1.0
    result = divergence(cell_dict, 13, dx, dy, dz)
    assert isinstance(result, float)
    assert abs(result) < 10.0



