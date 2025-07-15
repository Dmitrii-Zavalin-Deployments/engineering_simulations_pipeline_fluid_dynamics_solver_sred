# tests/test_pressure_utils.py
# 🧪 Unit tests for pressure utils — verifies indexing, mapping, and flattening of pressure fields

from src.physics.pressure_methods.utils import (
    index_fluid_cells,
    build_pressure_map,
    flatten_pressure_field
)
from src.grid_modules.cell import Cell

def make_cell(x, y, z, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=pressure, fluid_mask=fluid)

def test_index_fluid_cells_returns_coordinates_only():
    grid = [
        make_cell(0.0, 0.0, 0.0, 10.0, fluid=True),
        make_cell(1.0, 0.0, 0.0, 20.0, fluid=False),
        make_cell(2.0, 0.0, 0.0, 30.0, fluid=True)
    ]
    coords = index_fluid_cells(grid)
    assert coords == [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

def test_index_fluid_cells_empty_grid():
    assert index_fluid_cells([]) == []

def test_build_pressure_map_includes_only_fluid_with_numeric_pressure():
    grid = [
        make_cell(0.0, 0.0, 0.0, 10.0, fluid=True),
        make_cell(1.0, 0.0, 0.0, "non-numeric", fluid=True),
        make_cell(2.0, 0.0, 0.0, None, fluid=True),
        make_cell(3.0, 0.0, 0.0, 99.0, fluid=False)
    ]
    pressure_map = build_pressure_map(grid)
    assert pressure_map == {(0.0, 0.0, 0.0): 10.0}

def test_build_pressure_map_empty_grid():
    assert build_pressure_map([]) == {}

def test_flatten_pressure_field_preserves_order():
    pressure_map = {
        (0.0, 0.0, 0.0): 10.0,
        (1.0, 0.0, 0.0): 20.0,
        (2.0, 0.0, 0.0): 30.0
    }
    fluid_coords = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    result = flatten_pressure_field(pressure_map, fluid_coords)
    assert result == [20.0, 10.0]

def test_flatten_pressure_field_missing_coord_defaults_to_zero():
    pressure_map = {(0.0, 0.0, 0.0): 10.0}
    fluid_coords = [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)]
    result = flatten_pressure_field(pressure_map, fluid_coords)
    assert result == [0.0, 10.0]

def test_flatten_pressure_field_empty_inputs():
    assert flatten_pressure_field({}, []) == []