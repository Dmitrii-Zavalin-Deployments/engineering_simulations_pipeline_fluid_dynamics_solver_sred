# tests/grid/test_grid_generator.py

import pytest
from src.grid_generator import generate_grid
from src.grid_modules.cell import Cell

def test_basic_grid_structure():
    domain = {
        "nx": 2, "ny": 2, "nz": 1,
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 0.1
    }
    initial_conditions = {
        "initial_velocity": [1.0, 0.0, 0.0],
        "initial_pressure": 1.0
    }

    grid = generate_grid(domain, initial_conditions)
    assert len(grid) == 4
    for cell in grid:
        assert isinstance(cell, Cell)
        assert isinstance(cell.x, (int, float))
        assert isinstance(cell.y, (int, float))
        assert isinstance(cell.z, (int, float))
        assert isinstance(cell.velocity, list)
        assert len(cell.velocity) == 3
        assert all(isinstance(v, (int, float)) for v in cell.velocity)
        assert isinstance(cell.pressure, (int, float))
        assert isinstance(cell.fluid_mask, bool)
        assert cell.fluid_mask is True

def test_empty_domain_returns_empty():
    domain = {
        "nx": 0, "ny": 0, "nz": 0,
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    initial_conditions = {
        "initial_velocity": [0.0, 0.0, 0.0],
        "initial_pressure": 0.0
    }

    with pytest.raises(ValueError):
        generate_grid(domain, initial_conditions)

def test_nonuniform_dimensions():
    domain = {
        "nx": 1, "ny": 3, "nz": 2,
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 3.0,
        "min_z": 0.0, "max_z": 2.0
    }
    initial_conditions = {
        "initial_velocity": [0.0, 1.0, 0.0],
        "initial_pressure": 2.0
    }

    grid = generate_grid(domain, initial_conditions)
    assert len(grid) == 6
    for cell in grid:
        assert cell.velocity == [0.0, 1.0, 0.0]
        assert cell.pressure == 2.0
        assert cell.fluid_mask is True

def test_velocity_and_pressure_assigned_correctly():
    domain = {
        "nx": 2, "ny": 2, "nz": 1,
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 0.1
    }
    initial_conditions = {
        "initial_velocity": [3.0, -1.0, 0.5],
        "initial_pressure": 4.2
    }

    grid = generate_grid(domain, initial_conditions)
    for cell in grid:
        assert cell.velocity == [3.0, -1.0, 0.5]
        assert cell.pressure == 4.2
        assert cell.fluid_mask is True

def test_invalid_initial_conditions_handled_gracefully():
    domain = {
        "nx": 1, "ny": 1, "nz": 1,
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    initial_conditions = {
        "initial_velocity": None,
        "initial_pressure": None
    }

    grid = generate_grid(domain, initial_conditions)
    assert len(grid) == 1
    cell = grid[0]
    assert isinstance(cell.velocity, list)
    assert len(cell.velocity) == 3
    assert cell.velocity == [0.0, 0.0, 0.0]
    assert isinstance(cell.pressure, (int, float))
    assert cell.pressure == 0.0
    assert cell.fluid_mask is True

def test_large_grid_scale():
    domain = {
        "nx": 10, "ny": 10, "nz": 10,
        "min_x": 0.0, "max_x": 10.0,
        "min_y": 0.0, "max_y": 10.0,
        "min_z": 0.0, "max_z": 10.0
    }
    initial_conditions = {
        "initial_velocity": [0.0, 0.0, 1.0],
        "initial_pressure": 1.0
    }

    grid = generate_grid(domain, initial_conditions)
    assert len(grid) == 1000
    for cell in grid:
        assert cell.velocity == [0.0, 0.0, 1.0]
        assert cell.pressure == 1.0
        assert cell.fluid_mask is True



