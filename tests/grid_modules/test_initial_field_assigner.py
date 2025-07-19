# tests/grid_modules/test_initial_field_assigner.py
# 🧪 Unit tests for src/grid_modules/initial_field_assigner.py

import pytest
from src.grid_modules.cell import Cell
from src.grid_modules.initial_field_assigner import assign_fields

def make_cell(x=0.0, y=0.0, z=0.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=fluid_mask)

def test_assign_fields_to_fluid_cells():
    cells = [make_cell(), make_cell()]
    conditions = {"initial_velocity": [1.0, 0.0, -1.0], "initial_pressure": 50.0}
    updated = assign_fields(cells, conditions)

    for cell in updated:
        assert cell.velocity == [1.0, 0.0, -1.0]
        assert cell.pressure == 50.0

def test_assign_fields_to_solid_cells_sets_none():
    cells = [make_cell(fluid_mask=False)]
    conditions = {"initial_velocity": [1.0, 1.0, 1.0], "initial_pressure": 10.0}
    updated = assign_fields(cells, conditions)

    assert updated[0].velocity is None
    assert updated[0].pressure is None

def test_assign_fields_defaults_to_fluid_if_mask_missing():
    cell = Cell(x=0, y=0, z=0, velocity=[0, 0, 0], pressure=0, fluid_mask=None)
    del cell.fluid_mask  # simulate missing attribute
    conditions = {"initial_velocity": [1.0, 2.0, 3.0], "initial_pressure": 25.0}
    updated = assign_fields([cell], conditions)

    assert updated[0].velocity == [1.0, 2.0, 3.0]
    assert updated[0].pressure == 25.0

def test_assign_fields_missing_velocity_key_raises():
    cells = [make_cell()]
    conditions = {"initial_pressure": 10.0}
    with pytest.raises(ValueError, match="Missing 'initial_velocity'"):
        assign_fields(cells, conditions)

def test_assign_fields_missing_pressure_key_raises():
    cells = [make_cell()]
    conditions = {"initial_velocity": [0.0, 1.0, 0.0]}
    with pytest.raises(ValueError, match="Missing 'initial_pressure'"):
        assign_fields(cells, conditions)

def test_assign_fields_invalid_velocity_format():
    cells = [make_cell()]
    conditions = {"initial_velocity": [1.0, "a", 3.0], "initial_pressure": 5.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(cells, conditions)

def test_assign_fields_invalid_velocity_length():
    cells = [make_cell()]
    conditions = {"initial_velocity": [1.0], "initial_pressure": 5.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(cells, conditions)

def test_assign_fields_invalid_pressure_type():
    cells = [make_cell()]
    conditions = {"initial_velocity": [1.0, 2.0, 3.0], "initial_pressure": "high"}
    with pytest.raises(ValueError, match="must be a numeric value"):
        assign_fields(cells, conditions)



