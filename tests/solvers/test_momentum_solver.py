# tests/test_momentum_solver.py
# 🧪 Validates momentum solver: velocity evolution via advection + viscosity, fluid-only updates

from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update
import pytest

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

@pytest.fixture
def input_data():
    return {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "fluid_properties": {"viscosity": 0.5},
        "boundary_conditions": {
            "apply_faces": ["x_min", "x_max"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0
        }
    }

def test_momentum_evolves_fluid_velocity(input_data):
    cell = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], 10.0)
    result = apply_momentum_update([cell], input_data, step=0)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].fluid_mask is True
    assert result[0].velocity != [1.0, 0.0, 0.0]  # updated
    assert result[0].pressure == 10.0  # preserved

def test_momentum_preserves_nonfluid_cells(input_data):
    cell = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], 5.0, fluid=False)
    result = apply_momentum_update([cell], input_data, step=1)
    assert result[0].fluid_mask is False
    assert result[0].velocity is None
    assert result[0].pressure is None

def test_momentum_updates_multiple_cells(input_data):
    f1 = make_cell(0.0, 0.0, 0.0, [0.5, 0.0, 0.0], 1.0)
    f2 = make_cell(1.0, 0.0, 0.0, [1.5, 0.0, 0.0], 2.0)
    s1 = make_cell(2.0, 0.0, 0.0, [2.5, 0.0, 0.0], 3.0, fluid=False)
    grid = [f1, f2, s1]
    result = apply_momentum_update(grid, input_data, step=2)
    assert len(result) == 3
    assert result[0].fluid_mask is True
    assert result[1].fluid_mask is True
    assert result[2].fluid_mask is False
    assert result[2].velocity is None

def test_momentum_handles_missing_velocity_safely(input_data):
    broken = Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True)
    result = apply_momentum_update([broken], input_data, step=3)
    assert result[0].velocity is None

def test_momentum_preserves_pressure_field(input_data):
    cell = make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0], pressure=42.0)
    result = apply_momentum_update([cell], input_data, step=4)
    assert result[0].pressure == 42.0

def test_momentum_returns_updated_velocity_shape(input_data):
    cell = make_cell(1.0, 0.0, 0.0, [2.0, 2.0, 2.0], pressure=1.0)
    result = apply_momentum_update([cell], input_data, step=5)
    assert isinstance(result[0].velocity, list)
    assert len(result[0].velocity) == 3