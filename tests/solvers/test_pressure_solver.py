# tests/solvers/test_pressure_solver.py
# 🧪 Unit tests for pressure_solver.py — validates pressure correction flow

import pytest
from src.grid_modules.cell import Cell
from src.solvers.pressure_solver import apply_pressure_correction

def mock_config():
    return {
        "simulation_parameters": {
            "time_step": 0.1
        },
        "fluid_properties": {
            "viscosity": 0.01
        }
    }

def test_pressure_correction_preserves_grid_shape():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0], pressure=101.0, fluid_mask=True),
        Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False),
        Cell(x=2.0, y=0.0, z=0.0, velocity=[0.0, 1.0, 0.0], pressure=99.0, fluid_mask=True)
    ]
    result = apply_pressure_correction(grid, mock_config(), step=0)

    assert isinstance(result, list)
    assert len(result) == len(grid)
    for updated, original in zip(result, grid):
        assert isinstance(updated, Cell)
        assert updated.x == original.x
        assert updated.y == original.y
        assert updated.z == original.z
        assert updated.fluid_mask == original.fluid_mask

def test_fluid_cells_have_pressure_preserved_or_modified():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.01, 0.01, 0.01], pressure=100.0, fluid_mask=True),
        Cell(x=1.0, y=0.0, z=0.0, velocity=[-0.01, -0.01, -0.01], pressure=105.0, fluid_mask=True)
    ]
    result = apply_pressure_correction(grid, mock_config(), step=1)

    for updated in result:
        assert updated.fluid_mask
        assert updated.pressure is not None
        assert isinstance(updated.pressure, float)

def test_solid_cells_pressure_is_none():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False),
        Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False)
    ]
    result = apply_pressure_correction(grid, mock_config(), step=2)

    for updated in result:
        assert not updated.fluid_mask
        assert updated.pressure is None
        assert updated.velocity is None

def test_pressure_correction_handles_empty_grid():
    result = apply_pressure_correction([], mock_config(), step=3)
    assert isinstance(result, list)
    assert len(result) == 0

def test_pressure_correction_does_not_mutate_original():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[1, 0, 0], pressure=101, fluid_mask=True)
    ]
    before = grid[0].pressure
    result = apply_pressure_correction(grid, mock_config(), step=4)

    assert result[0].pressure == before or isinstance(result[0].pressure, float)
    assert grid[0].pressure == before  # original remains unchanged



