# tests/physics/pressure_methods/test_utils.py
# 🧪 Unit tests for pressure solver utilities

import pytest
from src.grid_modules.cell import Cell
from src.physics.pressure_methods.utils import (
    index_fluid_cells,
    build_pressure_map,
    flatten_pressure_field
)

def make_cell(x, y, z, pressure=0.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=pressure, fluid_mask=fluid_mask)

# ------------------------------
# index_fluid_cells tests
# ------------------------------

def test_index_fluid_cells_identifies_only_fluid():
    grid = [
        make_cell(0, 0, 0, fluid_mask=True),
        make_cell(1, 0, 0, fluid_mask=False),
        make_cell(2, 0, 0, fluid_mask=True)
    ]
    coords = index_fluid_cells(grid)
    assert coords == [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

def test_index_fluid_cells_empty_grid():
    coords = index_fluid_cells([])
    assert coords == []

def test_index_fluid_cells_all_solid():
    grid = [make_cell(0, 0, 0, fluid_mask=False), make_cell(1, 1, 1, fluid_mask=False)]
    coords = index_fluid_cells(grid)
    assert coords == []

# ------------------------------
# build_pressure_map tests
# ------------------------------

def test_build_pressure_map_includes_fluid_cells_only():
    grid = [
        make_cell(0, 0, 0, pressure=10.0, fluid_mask=True),
        make_cell(1, 0, 0, pressure=20.0, fluid_mask=False),
        make_cell(2, 0, 0, pressure=30.0, fluid_mask=True)
    ]
    pressure_map = build_pressure_map(grid)
    assert pressure_map == {
        (0.0, 0.0, 0.0): 10.0,
        (2.0, 0.0, 0.0): 30.0
    }

def test_build_pressure_map_skips_non_numeric_pressure():
    grid = [
        make_cell(0, 0, 0, pressure="bad", fluid_mask=True),
        make_cell(1, 0, 0, pressure=None, fluid_mask=True),
        make_cell(2, 0, 0, pressure=25.0, fluid_mask=True)
    ]
    pressure_map = build_pressure_map(grid)
    assert pressure_map == {(2.0, 0.0, 0.0): 25.0}

def test_build_pressure_map_all_solid():
    grid = [make_cell(0, 0, 0, pressure=100.0, fluid_mask=False)]
    pressure_map = build_pressure_map(grid)
    assert pressure_map == {}

# ------------------------------
# flatten_pressure_field tests
# ------------------------------

def test_flatten_pressure_field_matches_coord_order():
    pressure_map = {
        (0.0, 0.0, 0.0): 10.0,
        (1.0, 0.0, 0.0): 20.0,
        (2.0, 0.0, 0.0): 30.0
    }
    fluid_coords = [(2.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    result = flatten_pressure_field(pressure_map, fluid_coords)
    assert result == [30.0, 10.0]

def test_flatten_pressure_field_missing_coord_returns_zero():
    pressure_map = {
        (1.0, 0.0, 0.0): 50.0
    }
    fluid_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    result = flatten_pressure_field(pressure_map, fluid_coords)
    assert result == [0.0, 50.0]

def test_flatten_pressure_field_empty_inputs():
    assert flatten_pressure_field({}, []) == []



