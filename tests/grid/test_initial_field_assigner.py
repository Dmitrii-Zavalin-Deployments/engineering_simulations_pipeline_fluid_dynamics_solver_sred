# tests/grid/test_initial_field_assigner.py
# 🧪 Unit tests for initial_field_assigner — validates velocity and pressure assignment based on fluid mask

import pytest
from src.grid_modules.cell import Cell
from src.grid_modules.initial_field_assigner import assign_fields

def make_cell(fluid_mask=True):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=fluid_mask)

def test_valid_assignment_to_fluid_cells():
    cells = [make_cell(True), make_cell(True)]
    init = {"initial_velocity": [1.0, 0.0, 0.0], "initial_pressure": 50.0}
    result = assign_fields(cells, init)
    for cell in result:
        assert cell.velocity == [1.0, 0.0, 0.0]
        assert cell.pressure == 50.0

def test_valid_assignment_skips_solid_cells():
    cells = [make_cell(False), make_cell(True)]
    init = {"initial_velocity": [0.1, 0.2, 0.3], "initial_pressure": 42.0}
    result = assign_fields(cells, init)
    assert result[0].velocity is None
    assert result[0].pressure is None
    assert result[1].velocity == [0.1, 0.2, 0.3]
    assert result[1].pressure == 42.0

def test_fluid_mask_missing_defaults_to_true():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=None)
    del cell.fluid_mask  # Remove attribute entirely
    init = {"initial_velocity": [0.0, 0.0, 1.0], "initial_pressure": 25.0}
    result = assign_fields([cell], init)
    assert result[0].velocity == [0.0, 0.0, 1.0]
    assert result[0].pressure == 25.0

def test_missing_initial_velocity_raises():
    init = {"initial_pressure": 100.0}
    cells = [make_cell(True)]
    with pytest.raises(ValueError, match="Missing 'initial_velocity'"):
        assign_fields(cells, init)

def test_missing_initial_pressure_raises():
    init = {"initial_velocity": [0.0, 1.0, 0.0]}
    cells = [make_cell(True)]
    with pytest.raises(ValueError, match="Missing 'initial_pressure'"):
        assign_fields(cells, init)

def test_non_numeric_pressure_raises():
    init = {"initial_velocity": [0.0, 0.0, 1.0], "initial_pressure": "high"}
    cells = [make_cell(True)]
    with pytest.raises(ValueError, match="must be a numeric value"):
        assign_fields(cells, init)

@pytest.mark.parametrize("bad_velocity", [
    "string", [1.0, 2.0], [1.0, "fast", 0.0], None
])
def test_invalid_velocity_formats_raise(bad_velocity):
    init = {"initial_velocity": bad_velocity, "initial_pressure": 10.0}
    cells = [make_cell(True)]
    with pytest.raises(ValueError, match="must be a list of 3 numeric"):
        assign_fields(cells, init)

def test_empty_cell_list_returns_empty():
    init = {"initial_velocity": [0.0, 0.0, 0.0], "initial_pressure": 10.0}
    result = assign_fields([], init)
    assert result == []

def test_multiple_cell_masks_applied_correctly():
    cells = [make_cell(True), make_cell(False), make_cell(True)]
    init = {"initial_velocity": [1.0, 1.0, 1.0], "initial_pressure": 10.0}
    result = assign_fields(cells, init)
    assert result[0].velocity == [1.0, 1.0, 1.0]
    assert result[1].velocity is None
    assert result[2].pressure == 10.0