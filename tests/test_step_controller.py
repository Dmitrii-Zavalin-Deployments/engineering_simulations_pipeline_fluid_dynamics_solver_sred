# tests/test_step_controller.py
# 🧪 Tests for simulation step controller — verifies grid evolution and reflex metadata

import pytest
from src.step_controller import evolve_step
from src.grid_modules.cell import Cell

def make_fluid_cell(x, y, z, velocity=None, pressure=10.0):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity if velocity is not None else [1.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=True
    )

def make_solid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)

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
            "nx": 10,
            "min_y": 0.0,
            "max_y": 1.0,
            "ny": 1,
            "min_z": 0.0,
            "max_z": 1.0,
            "nz": 1
        }
    }

def test_evolve_single_fluid_cell():
    grid = [make_fluid_cell(0, 0, 0)]
    updated, reflex = evolve_step(grid, mock_config(), step=0)
    assert len(updated) == 1
    cell = updated[0]
    assert cell.fluid_mask
    assert isinstance(cell.velocity, list)
    assert isinstance(cell.pressure, float)
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex

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
            assert isinstance(cell.pressure, float)
        else:
            assert cell.velocity is None
            assert cell.pressure is None
    assert "max_velocity" in reflex

def test_evolve_empty_grid():
    updated, reflex = evolve_step([], mock_config(), step=2)
    assert updated == []
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex

def test_evolve_malformed_velocity_gets_downgraded_to_solid():
    bad_cell = Cell(x=0, y=0, z=0, velocity="bad", pressure=1.0, fluid_mask=True)
    grid = [bad_cell]
    updated, reflex = evolve_step(grid, mock_config(), step=3)
    assert len(updated) == 1
    downgraded = updated[0]
    assert downgraded.fluid_mask is False
    assert downgraded.velocity is None
    assert downgraded.pressure is None
    assert isinstance(reflex["max_velocity"], float)

def test_reflex_metadata_structure():
    grid = [make_fluid_cell(0, 0, 0)]
    _, reflex = evolve_step(grid, mock_config(), step=4)
    expected_keys = {
        "max_velocity",
        "global_cfl",
        "max_divergence",
        "damping_enabled",
        "overflow_detected",
        "adjusted_time_step",
        "projection_passes"
    }
    for key in expected_keys:
        assert key in reflex

def test_evolve_step_index_effect():
    grid = [make_fluid_cell(0, 0, 0)]
    updated_a, reflex_a = evolve_step(grid, mock_config(), step=0)
    updated_b, reflex_b = evolve_step(grid, mock_config(), step=1)
    assert len(updated_a) == len(updated_b)
    assert isinstance(reflex_a, dict)
    assert isinstance(reflex_b, dict)
    assert set(reflex_a.keys()) == set(reflex_b.keys())



