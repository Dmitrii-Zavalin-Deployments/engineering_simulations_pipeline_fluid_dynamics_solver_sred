# tests/physics/pressure_methods/test_utils.py
# ✅ Validation suite for src/physics/pressure_methods/utils.py

import pytest
from src.physics.pressure_methods.utils import index_fluid_cells, build_pressure_map
from src.grid_modules.cell import Cell

def make_cell(x, y, z, pressure=None, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=[0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )

# 🔍 index_fluid_cells tests
def test_index_fluid_cells_returns_only_fluid_coords():
    grid = [
        make_cell(0.0, 0.0, 0.0, fluid_mask=True),
        make_cell(1.0, 0.0, 0.0, fluid_mask=False),
        make_cell(0.0, 1.0, 0.0, fluid_mask=True)
    ]
    result = index_fluid_cells(grid)
    assert result == [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

def test_index_fluid_cells_returns_empty_on_all_solid():
    grid = [
        make_cell(0.0, 0.0, 0.0, fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, fluid_mask=False)
    ]
    result = index_fluid_cells(grid)
    assert result == []

def test_index_fluid_cells_handles_empty_grid():
    assert index_fluid_cells([]) == []

# 🔍 build_pressure_map tests
def test_build_pressure_map_includes_valid_pressures():
    grid = [
        make_cell(0.0, 0.0, 0.0, pressure=5.0),
        make_cell(1.0, 0.0, 0.0, pressure=3.2),
        make_cell(0.0, 1.0, 0.0, pressure=None)
    ]
    result = build_pressure_map(grid)
    assert result == {
        (0.0, 0.0, 0.0): 5.0,
        (1.0, 0.0, 0.0): 3.2
    }

def test_build_pressure_map_excludes_non_numeric_pressures():
    grid = [
        make_cell(0.0, 0.0, 0.0, pressure="invalid"),
        make_cell(1.0, 0.0, 0.0, pressure=None),
        make_cell(2.0, 0.0, 0.0, pressure=0.0)
    ]
    result = build_pressure_map(grid)
    assert result == {(2.0, 0.0, 0.0): 0.0}

def test_build_pressure_map_handles_empty_grid():
    assert build_pressure_map([]) == {}



