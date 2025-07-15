# tests/physics/test_velocity_projection.py
# 🧪 Validates pressure-gradient velocity adjustment for fluid incompressibility enforcement

from src.grid_modules.cell import Cell
from src.physics.velocity_projection import apply_pressure_velocity_projection
import pytest

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

@pytest.fixture
def config_3x1x1():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }

def test_projection_subtracts_pressure_gradient(config_3x1x1):
    dx = 1.0  # from domain definition
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], 10.0),
        make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0], 20.0),  # center
        make_cell(2.0, 0.0, 0.0, [0.0, 0.0, 0.0], 30.0)
    ]
    projected = apply_pressure_velocity_projection(grid, config_3x1x1)
    expected_grad_x = (30.0 - 10.0) / (2 * dx)  # = 10.0
    expected_velocity = [1.0 - expected_grad_x, 1.0, 1.0]
    assert projected[1].velocity == pytest.approx(expected_velocity)

def test_projection_skips_missing_pressure_neighbors(config_3x1x1):
    grid = [
        make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0], 10.0),
        make_cell(2.0, 0.0, 0.0, [1.0, 1.0, 1.0], None)
    ]
    projected = apply_pressure_velocity_projection(grid, config_3x1x1)
    assert projected[0].velocity == [1.0, 1.0, 1.0]  # unchanged
    assert projected[1].velocity == [1.0, 1.0, 1.0]  # skipped gradient logic due to missing pressure

def test_projection_skips_nonfluid_cells(config_3x1x1):
    solid = make_cell(1.0, 0.0, 0.0, [2.0, 2.0, 2.0], 99.0, fluid=False)
    result = apply_pressure_velocity_projection([solid], config_3x1x1)
    assert result[0].velocity == [2.0, 2.0, 2.0]  # preserved

def test_projection_requires_velocity_and_pressure(config_3x1x1):
    cell = Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=True)
    result = apply_pressure_velocity_projection([cell], config_3x1x1)
    assert result[0].velocity is None
    assert result[0].pressure is None

def test_projection_applies_all_axis_gradients(config_3x1x1):
    grid = [
        make_cell(1.0, 0.0, 1.0, [1.0, 1.0, 1.0], 20.0),
        make_cell(0.0, 0.0, 1.0, [0.0, 0.0, 0.0], 10.0),
        make_cell(2.0, 0.0, 1.0, [0.0, 0.0, 0.0], 30.0),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], 10.0),
        make_cell(1.0, 0.0, 2.0, [0.0, 0.0, 0.0], 30.0),
    ]
    result = apply_pressure_velocity_projection(grid, config_3x1x1)
    # Central difference for x and z: grad_x = (30-10)/2 = 10; grad_z = (30-10)/2 = 10
    expected = [1.0 - 10.0, 1.0, 1.0 - 10.0]
    assert result[0].velocity == pytest.approx(expected)

def test_projection_handles_partial_neighbors(config_3x1x1):
    grid = [
        make_cell(1.0, 0.0, 1.0, [1.0, 1.0, 1.0], 20.0),
        make_cell(2.0, 0.0, 1.0, [0.0, 0.0, 0.0], 30.0)
    ]
    result = apply_pressure_velocity_projection(grid, config_3x1x1)
    assert result[0].velocity == [1.0, 1.0, 1.0]  # no gradient applied

def test_projection_returns_updated_grid(config_3x1x1):
    fluid = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], 10.0)
    result = apply_pressure_velocity_projection([fluid], config_3x1x1)
    assert isinstance(result, list)
    assert result[0].fluid_mask is True
    assert isinstance(result[0].velocity, list)



