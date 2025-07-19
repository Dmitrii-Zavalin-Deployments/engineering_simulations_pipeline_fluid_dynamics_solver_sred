# tests/physics/pressure_methods/test_utils.py
# 🧪 Unit tests for src/physics/pressure_methods/utils.py

from src.grid_modules.cell import Cell
from src.physics.pressure_methods.utils import (
    index_fluid_cells,
    build_pressure_map,
    flatten_pressure_field
)

def make_cell(x, y, z, pressure, fluid):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=pressure, fluid_mask=fluid)

def test_index_fluid_cells_returns_only_fluid_coords():
    grid = [
        make_cell(1, 2, 3, pressure=5.0, fluid=True),
        make_cell(4, 5, 6, pressure=3.0, fluid=False)
    ]
    result = index_fluid_cells(grid)
    assert result == [(1, 2, 3)]

def test_build_pressure_map_skips_non_fluid():
    grid = [
        make_cell(1, 1, 1, pressure=2.0, fluid=True),
        make_cell(2, 2, 2, pressure=4.0, fluid=False)
    ]
    result = build_pressure_map(grid)
    assert (1, 1, 1) in result
    assert (2, 2, 2) not in result

def test_build_pressure_map_skips_non_numeric():
    grid = [
        make_cell(1, 1, 1, pressure="hi", fluid=True),
        make_cell(2, 2, 2, pressure=None, fluid=True)
    ]
    result = build_pressure_map(grid)
    assert result == {}

def test_flatten_pressure_field_returns_values_in_given_order():
    pressure_map = {
        (0, 0, 0): 5.0,
        (1, 0, 0): 3.0,
        (2, 0, 0): 4.5
    }
    coords = [(2, 0, 0), (0, 0, 0), (1, 0, 0)]
    result = flatten_pressure_field(pressure_map, coords)
    assert result == [4.5, 5.0, 3.0]

def test_flatten_pressure_field_defaults_missing_to_zero():
    pressure_map = {(0, 0, 0): 1.0}
    coords = [(0, 0, 0), (9, 9, 9)]
    result = flatten_pressure_field(pressure_map, coords)
    assert result == [1.0, 0.0]



