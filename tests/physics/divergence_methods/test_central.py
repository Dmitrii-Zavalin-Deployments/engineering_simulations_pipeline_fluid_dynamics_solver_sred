# tests/physics/divergence_methods/test_central.py
# 🧪 Unit tests for compute_central_divergence — validates central differencing for fluid grids

import pytest
from src.grid_modules.cell import Cell
from src.physics.divergence_methods.central import compute_central_divergence

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid)

@pytest.fixture
def config_3x1x1():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }

def test_single_fluid_cell_returns_zero(config_3x1x1):
    grid = [make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0])]
    result = compute_central_divergence(grid, config_3x1x1)
    assert result == [0.0]

def test_skips_solid_cells(config_3x1x1):
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid=True),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid=True),
        make_cell(2.0, 0.0, 0.0, [2.0, 0.0, 0.0], fluid=True)
    ]
    result = compute_central_divergence(grid, config_3x1x1)
    assert result[1] == pytest.approx((2.0 - 0.0) / (2.0 * 1.0))  # dx = 1.0

def test_full_x_axis_gradient(config_3x1x1):
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    ]
    result = compute_central_divergence(grid, config_3x1x1)
    assert result[1] == pytest.approx(1.0)

def test_y_axis_contribution(config_3x1x1):
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1,
            "min_y": 0.0, "max_y": 2.0, "ny": 2,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(0.0, 1.0, 0.0, [0.0, 2.0, 0.0]),
        make_cell(0.0, 2.0, 0.0, [0.0, 4.0, 0.0])
    ]
    result = compute_central_divergence(grid, config)
    assert result[1] == pytest.approx(2.0)

def test_z_axis_contribution(config_3x1x1):
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 2.0, "nz": 2
        }
    }
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(0.0, 0.0, 1.0, [0.0, 0.0, 2.0]),
        make_cell(0.0, 0.0, 2.0, [0.0, 0.0, 4.0])
    ]
    result = compute_central_divergence(grid, config)
    assert result[1] == pytest.approx(2.0)

def test_velocity_missing_or_malformed_skipped(config_3x1x1):
    grid = [
        make_cell(1.0, 0.0, 0.0, None),
        make_cell(2.0, 0.0, 0.0, [1.0, 2.0])  # Not 3D
    ]
    result = compute_central_divergence(grid, config_3x1x1)
    assert result == []

def test_multiple_valid_fluid_cells(config_3x1x1):
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0]),
        make_cell(2.0, 0.0, 0.0, [2.0, 2.0, 2.0])
    ]
    result = compute_central_divergence(grid, config_3x1x1)
    assert result[1] == pytest.approx(3.0)



