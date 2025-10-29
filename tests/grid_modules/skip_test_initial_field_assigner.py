# tests/grid_modules/test_initial_field_assigner.py
# ✅ Validation suite for src/grid_modules/initial_field_assigner.py

import pytest
from src.grid_modules.initial_field_assigner import assign_fields
from src.grid_modules.cell import Cell

def mock_cell(x=0, y=0, z=0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=fluid_mask)

def test_assign_fields_to_fluid_cells_only():
    cells = [
        mock_cell(0, 0, 0, fluid_mask=True),
        mock_cell(1, 1, 1, fluid_mask=False)
    ]
    initial_conditions = {
        "initial_velocity": [1.0, 2.0, 3.0],
        "initial_pressure": 101325.0
    }

    result = assign_fields(cells, initial_conditions)
    assert result[0].velocity == [1.0, 2.0, 3.0]
    assert result[0].pressure == 101325.0
    assert result[1].velocity is None
    assert result[1].pressure is None

def test_assign_fields_missing_velocity_key():
    cells = [mock_cell()]
    initial_conditions = {
        "initial_pressure": 101325.0
    }

    with pytest.raises(ValueError) as e:
        assign_fields(cells, initial_conditions)
    assert "Missing 'initial_velocity'" in str(e.value)

def test_assign_fields_missing_pressure_key():
    cells = [mock_cell()]
    initial_conditions = {
        "initial_velocity": [1.0, 2.0, 3.0]
    }

    with pytest.raises(ValueError) as e:
        assign_fields(cells, initial_conditions)
    assert "Missing 'initial_pressure'" in str(e.value)

def test_assign_fields_invalid_velocity_type():
    cells = [mock_cell()]
    initial_conditions = {
        "initial_velocity": "not a list",
        "initial_pressure": 101325.0
    }

    with pytest.raises(ValueError) as e:
        assign_fields(cells, initial_conditions)
    assert "'initial_velocity' must be a list" in str(e.value)

def test_assign_fields_invalid_velocity_length():
    cells = [mock_cell()]
    initial_conditions = {
        "initial_velocity": [1.0, 2.0],  # too short
        "initial_pressure": 101325.0
    }

    with pytest.raises(ValueError) as e:
        assign_fields(cells, initial_conditions)
    assert "'initial_velocity' must be a list of 3" in str(e.value)

def test_assign_fields_invalid_velocity_components():
    cells = [mock_cell()]
    initial_conditions = {
        "initial_velocity": [1.0, "bad", 3.0],
        "initial_pressure": 101325.0
    }

    with pytest.raises(ValueError) as e:
        assign_fields(cells, initial_conditions)
    assert "must be a list of 3 numeric components" in str(e.value)

def test_assign_fields_invalid_pressure_type():
    cells = [mock_cell()]
    initial_conditions = {
        "initial_velocity": [1.0, 2.0, 3.0],
        "initial_pressure": "not a number"
    }

    with pytest.raises(ValueError) as e:
        assign_fields(cells, initial_conditions)
    assert "'initial_pressure' must be a numeric value" in str(e.value)

def test_assign_fields_mutates_cells_in_place():
    cell = mock_cell()
    initial_conditions = {
        "initial_velocity": [0.0, 0.0, 0.0],
        "initial_pressure": 0.0
    }

    assign_fields([cell], initial_conditions)
    assert cell.velocity == [0.0, 0.0, 0.0]
    assert cell.pressure == 0.0



