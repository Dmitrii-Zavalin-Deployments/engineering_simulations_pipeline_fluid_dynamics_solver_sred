# tests/grid/test_initial_field_assigner.py

import pytest
from src.grid_modules.cell import Cell
from src.grid_modules.initial_field_assigner import assign_fields

# 🔧 Helper: generate basic grid
def make_cells(count=3):
    return [Cell(x=i, y=0, z=0, velocity=[], pressure=0.0, fluid_mask=True) for i in range(count)]

# ✅ Test: Correct assignment
def test_valid_assignment():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 0.0, -1.0], "initial_pressure": 42.0}
    updated = assign_fields(grid, init)
    for cell in updated:
        assert cell.velocity == [1.0, 0.0, -1.0]
        assert cell.pressure == 42.0

# ❌ Test: Missing 'initial_velocity'
def test_missing_velocity_key():
    grid = make_cells()
    init = {"initial_pressure": 1.0}
    with pytest.raises(ValueError, match="Missing 'initial_velocity'"):
        assign_fields(grid, init)

# ❌ Test: Missing 'initial_pressure'
def test_missing_pressure_key():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="Missing 'initial_pressure'"):
        assign_fields(grid, init)

# ❌ Test: Velocity not a list
def test_velocity_not_list():
    grid = make_cells()
    init = {"initial_velocity": "fast", "initial_pressure": 1.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(grid, init)

# ❌ Test: Velocity wrong size
def test_velocity_wrong_size():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 2.0], "initial_pressure": 1.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(grid, init)

# ❌ Test: Velocity with non-numeric values
def test_velocity_non_numeric():
    grid = make_cells()
    init = {"initial_velocity": [1.0, "bad", 0.0], "initial_pressure": 1.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(grid, init)

# ❌ Test: Pressure not numeric
def test_pressure_not_numeric():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 0.0, 0.0], "initial_pressure": "high"}
    with pytest.raises(ValueError, match="must be a numeric value"):
        assign_fields(grid, init)

# ✅ Edge Case: Negative pressure and velocity values
def test_negative_velocity_and_pressure():
    grid = make_cells()
    init = {"initial_velocity": [-1.0, -0.5, -2.0], "initial_pressure": -99.0}
    updated = assign_fields(grid, init)
    for cell in updated:
        assert cell.velocity == [-1.0, -0.5, -2.0]
        assert cell.pressure == -99.0



