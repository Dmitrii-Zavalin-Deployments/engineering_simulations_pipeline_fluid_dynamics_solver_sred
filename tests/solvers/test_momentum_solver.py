# tests/solvers/test_momentum_solver.py
# 🧪 Unit tests for momentum_solver.py — velocity evolution validation

import pytest
from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update

def mock_input_config():
    return {
        "simulation_parameters": {
            "time_step": 0.1
        },
        "fluid_properties": {
            "viscosity": 0.01
        }
    }

def test_velocity_evolution_preserves_structure():
    initial_grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.01, 0.0, 0.0], pressure=100.0, fluid_mask=True),
        Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False),
        Cell(x=2.0, y=0.0, z=0.0, velocity=[0.01, 0.01, 0.0], pressure=95.0, fluid_mask=True)
    ]
    updated_grid = apply_momentum_update(initial_grid, mock_input_config(), step=1)

    assert isinstance(updated_grid, list)
    assert len(updated_grid) == len(initial_grid)
    for updated, original in zip(updated_grid, initial_grid):
        assert isinstance(updated, Cell)
        assert updated.x == original.x
        assert updated.y == original.y
        assert updated.z == original.z
        assert updated.fluid_mask == original.fluid_mask

def test_fluid_cells_preserve_pressure():
    initial_grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.02, 0.01, 0.0], pressure=99.0, fluid_mask=True),
        Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False),
    ]
    updated = apply_momentum_update(initial_grid, mock_input_config(), step=2)

    assert updated[0].pressure == initial_grid[0].pressure
    assert updated[1].pressure is None

def test_solid_cells_are_unchanged():
    initial_grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False),
        Cell(x=1.0, y=0.0, z=0.0, velocity=[0.05, 0.0, 0.0], pressure=90.0, fluid_mask=True),
    ]
    updated = apply_momentum_update(initial_grid, mock_input_config(), step=3)

    solid_cell = updated[0]
    assert solid_cell.velocity is None
    assert solid_cell.pressure is None
    assert solid_cell.fluid_mask is False

def test_velocity_list_preserved_for_fluid():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.01, 0.02, 0.03], pressure=98.0, fluid_mask=True),
    ]
    result = apply_momentum_update(grid, mock_input_config(), step=4)
    assert isinstance(result[0].velocity, list)
    assert len(result[0].velocity) == 3
    assert all(isinstance(v, float) for v in result[0].velocity)



