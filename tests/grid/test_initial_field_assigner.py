# tests/grid/test_initial_field_assigner.py

import pytest
from src.grid_modules.cell import Cell
from src.grid_modules.initial_field_assigner import assign_fields

# 🔧 Helper: generate basic grid
def make_cells(count=3, fluid_mask=True):
    return [Cell(x=i, y=0, z=0, velocity=[], pressure=0.0, fluid_mask=fluid_mask) for i in range(count)]

# ✅ Test: Correct assignment to fluid cells
def test_valid_assignment():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 0.0, -1.0], "initial_pressure": 42.0}
    updated = assign_fields(grid, init)
    for cell in updated:
        assert cell.velocity == [1.0, 0.0, -1.0]
        assert cell.pressure == 42.0

# ✅ Test: Solid cells remain uninitialized
def test_solid_cells_are_sanitized():
    fluid_cells = make_cells(2, fluid_mask=True)
    solid_cells = make_cells(2, fluid_mask=False)
    grid = fluid_cells + solid_cells
    init = {"initial_velocity": [2.0, 0.0, 0.0], "initial_pressure": 5.5}
    updated = assign_fields(grid, init)

    for cell in updated:
        if getattr(cell, "fluid_mask", True):
            assert cell.velocity == [2.0, 0.0, 0.0]
            assert cell.pressure == 5.5
        else:
            assert cell.velocity is None
            assert cell.pressure is None

# ✅ Test: Mixed fluid and solid cell indexing
def test_mixed_order_cell_assignment():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[], pressure=0.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[], pressure=0.0, fluid_mask=False),
        Cell(x=2, y=0, z=0, velocity=[], pressure=0.0, fluid_mask=True)
    ]
    init = {"initial_velocity": [0.5, 0.5, 0.0], "initial_pressure": 8.0}
    updated = assign_fields(grid, init)

    assert updated[0].velocity == [0.5, 0.5, 0.0]
    assert updated[0].pressure == 8.0
    assert updated[1].velocity is None
    assert updated[1].pressure is None
    assert updated[2].velocity == [0.5, 0.5, 0.0]
    assert updated[2].pressure == 8.0

# ✅ Test: Fallback for missing fluid_mask (treated as fluid)
def test_partial_cell_missing_fluid_mask_defaults_to_true():
    class PartialCell:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.velocity = []
            self.pressure = 0.0
            # fluid_mask intentionally omitted

    grid = [PartialCell(0, 0, 0), PartialCell(1, 0, 0)]
    init = {"initial_velocity": [1.0, 1.0, 1.0], "initial_pressure": 10.0}
    updated = assign_fields(grid, init)
    for cell in updated:
        assert cell.velocity == [1.0, 1.0, 1.0]
        assert cell.pressure == 10.0

# ❌ Test: Missing velocity key
def test_missing_initial_velocity_key():
    grid = make_cells()
    init = {"initial_pressure": 1.0}
    with pytest.raises(ValueError, match="Missing 'initial_velocity'"):
        assign_fields(grid, init)

# ❌ Test: Missing pressure key
def test_missing_initial_pressure_key():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="Missing 'initial_pressure'"):
        assign_fields(grid, init)

# ❌ Test: Velocity not a list
def test_velocity_is_not_list():
    grid = make_cells()
    init = {"initial_velocity": "fast", "initial_pressure": 1.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(grid, init)

# ❌ Test: Velocity incorrect length
def test_velocity_wrong_vector_size():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 2.0], "initial_pressure": 1.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(grid, init)

# ❌ Test: Velocity contains non-numeric values
def test_velocity_has_non_numeric():
    grid = make_cells()
    init = {"initial_velocity": [1.0, "bad", 0.0], "initial_pressure": 1.0}
    with pytest.raises(ValueError, match="must be a list of 3 numeric components"):
        assign_fields(grid, init)

# ❌ Test: Pressure is not numeric
def test_pressure_not_a_number():
    grid = make_cells()
    init = {"initial_velocity": [1.0, 0.0, 0.0], "initial_pressure": "high"}
    with pytest.raises(ValueError, match="must be a numeric value"):
        assign_fields(grid, init)

# ✅ Edge Case: Negative values
def test_negative_pressure_and_velocity_values():
    grid = make_cells()
    init = {"initial_velocity": [-1.0, -0.5, -2.0], "initial_pressure": -99.0}
    updated = assign_fields(grid, init)
    for cell in updated:
        assert cell.velocity == [-1.0, -0.5, -2.0]
        assert cell.pressure == -99.0

# 🔒 Defensive grid size check
def test_field_assigner_grid_size_consistency():
    grid = make_cells(count=6)
    expected_size = 6
    init = {"initial_velocity": [1.0, 0.0, 0.0], "initial_pressure": 100.0}
    updated = assign_fields(grid, init)
    assert len(updated) == expected_size, "❌ Grid size mismatch after field assignment"



