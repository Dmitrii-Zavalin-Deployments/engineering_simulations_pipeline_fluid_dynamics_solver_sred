# tests/test_step_controller.py
# 🧪 Tests for simulation step controller — verifies grid evolution and reflex metadata

import pytest
from src.step_controller import evolve_step
from src.grid_modules.cell import Cell

# 🔧 Minimal valid fluid cell
def make_fluid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=10.0, fluid_mask=True)

# 🔧 Minimal solid cell
def make_solid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)

# ✅ Updated mock config to include required domain_definition
def mock_config(time_step=0.1, viscosity=0.01):
    return {
        "simulation_parameters": {
            "time_step": time_step
        },
        "fluid_properties": {
            "viscosity": viscosity
        },
        "domain_definition": {
            "min_x": 0.0,
            "max_x": 1.0,
            "nx": 10
        }
    }

# ✅ Test: Evolve with single fluid cell
def test_evolve_single_fluid_cell():
    grid = [make_fluid_cell(0, 0, 0)]
    updated, reflex = evolve_step(grid, mock_config(), step=0)

    assert isinstance(updated, list)
    assert len(updated) == 1
    cell = updated[0]
    assert isinstance(cell.velocity, list)
    assert isinstance(cell.pressure, (int, float))
    assert cell.fluid_mask is True

    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex
    assert "damping_enabled" in reflex

# ✅ Test: Evolve with fluid and solid cells
def test_evolve_mixed_cells():
    grid = [
        make_fluid_cell(0, 0, 0),
        make_solid_cell(1, 0, 0),
        make_fluid_cell(2, 0, 0)
    ]
    updated, reflex = evolve_step(grid, mock_config(), step=1)

    assert len(updated) == 3
    for cell in updated:
        if cell.fluid_mask:
            assert isinstance(cell.velocity, list)
            assert isinstance(cell.pressure, (int, float))
        else:
            assert cell.velocity is None
            assert cell.pressure is None
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex

# ✅ Test: Evolve with empty grid
def test_evolve_empty_grid():
    grid = []
    updated, reflex = evolve_step(grid, mock_config(), step=2)

    assert updated == []
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex

# ✅ Test: Evolve with malformed velocity (should preserve or ignore)
def test_evolve_malformed_velocity():
    bad_cell = Cell(x=0, y=0, z=0, velocity="bad", pressure=1.0, fluid_mask=True)
    grid = [bad_cell]
    updated, reflex = evolve_step(grid, mock_config(), step=3)

    assert updated[0].fluid_mask is True
    assert isinstance(updated[0].pressure, (int, float))
    assert updated[0].velocity == "bad"
    assert isinstance(reflex["max_velocity"], float)

# ✅ Test: Reflex metadata structure is complete
def test_reflex_metadata_keys_present():
    grid = [make_fluid_cell(0, 0, 0)]
    _, reflex = evolve_step(grid, mock_config(), step=4)

    expected_keys = {
        "max_velocity",
        "max_divergence",
        "global_cfl",
        "overflow_detected",
        "damping_enabled",
        "adjusted_time_step",
        "projection_passes"
    }
    assert expected_keys.issubset(reflex.keys())

# ✅ Test: Step index variation preserves behavior
def test_evolve_step_index_variation():
    grid = [make_fluid_cell(0, 0, 0)]
    updated1, reflex1 = evolve_step(grid, mock_config(), step=0)
    updated2, reflex2 = evolve_step(grid, mock_config(), step=10)

    assert len(updated1) == len(updated2)
    assert isinstance(reflex1, dict)
    assert isinstance(reflex2, dict)
    assert reflex1.keys() == reflex2.keys()



