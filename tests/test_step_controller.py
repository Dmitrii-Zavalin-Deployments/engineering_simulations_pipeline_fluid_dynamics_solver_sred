# tests/test_step_controller.py

import pytest
from src.step_controller import evolve_step
from src.grid_modules.cell import Cell

# 🔧 Minimal valid fluid cell
def make_fluid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=10.0, fluid_mask=True)

# 🔧 Minimal solid cell
def make_solid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)

# ✅ Test: Evolve with single fluid cell
def test_evolve_single_fluid_cell():
    grid = [make_fluid_cell(0, 0, 0)]
    input_data = {"solver_params": {}, "reflex_flags": {}}
    updated = evolve_step(grid, input_data, step=0)
    assert isinstance(updated, list)
    assert len(updated) == 1
    cell = updated[0]
    assert isinstance(cell.velocity, list)
    assert isinstance(cell.pressure, (int, float))
    assert cell.fluid_mask is True

# ✅ Test: Evolve with fluid and solid cells
def test_evolve_mixed_cells():
    grid = [
        make_fluid_cell(0, 0, 0),
        make_solid_cell(1, 0, 0),
        make_fluid_cell(2, 0, 0)
    ]
    input_data = {"solver_params": {}, "reflex_flags": {}}
    updated = evolve_step(grid, input_data, step=1)
    assert len(updated) == 3
    for cell in updated:
        if cell.fluid_mask:
            assert isinstance(cell.velocity, list)
            assert isinstance(cell.pressure, (int, float))
        else:
            assert cell.velocity is None
            assert cell.pressure is None

# ✅ Test: Evolve with empty grid
def test_evolve_empty_grid():
    grid = []
    input_data = {"solver_params": {}, "reflex_flags": {}}
    updated = evolve_step(grid, input_data, step=2)
    assert updated == []

# ✅ Test: Evolve with malformed velocity (should preserve or ignore)
def test_evolve_malformed_velocity():
    bad_cell = Cell(x=0, y=0, z=0, velocity="bad", pressure=1.0, fluid_mask=True)
    grid = [bad_cell]
    input_data = {"solver_params": {}, "reflex_flags": {}}
    updated = evolve_step(grid, input_data, step=3)
    assert updated[0].fluid_mask is True
    assert isinstance(updated[0].pressure, (int, float))
    # Let broken velocity pass through unchanged for now
    assert updated[0].velocity == "bad"

# ✅ Test: Input data is passed correctly
def test_evolve_input_data_passthrough():
    grid = [make_fluid_cell(0, 0, 0)]
    input_data = {
        "solver_params": {
            "viscosity": 0.01,
            "external_force": [0.0, -9.8, 0.0]
        },
        "reflex_flags": {
            "damping_enabled": False
        }
    }
    updated = evolve_step(grid, input_data, step=5)
    assert len(updated) == 1
    assert updated[0].fluid_mask is True

# ✅ Test: Step index updates logging and doesn’t alter behavior
def test_evolve_step_index_variation():
    grid = [make_fluid_cell(0, 0, 0)]
    input_data = {"solver_params": {}, "reflex_flags": {}}
    before = evolve_step(grid, input_data, step=0)
    after = evolve_step(grid, input_data, step=10)
    assert len(before) == len(after)



