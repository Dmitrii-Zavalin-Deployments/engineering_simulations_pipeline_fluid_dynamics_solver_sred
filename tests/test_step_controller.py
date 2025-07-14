# tests/test_step_controller.py
# 🧪 Tests for simulation step controller — verifies grid evolution, reflex metadata, and ghost integration

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
            "min_x": 0.0, "max_x": 1.0, "nx": 5,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "boundary_conditions": {
            "x_min": "inlet",
            "x_max": "outlet",
            "y_min": "wall",
            "y_max": "wall",
            "z_min": "symmetry",
            "z_max": "symmetry",
            "faces": [1, 2, 3, 4, 5, 6],
            "type": "dirichlet",
            "apply_to": ["pressure", "velocity"],
            "pressure": 100.0,
            "velocity": [0.0, 0.0, 0.0],
            "no_slip": True
        }
    }

def test_evolve_single_fluid_cell():
    grid = [make_fluid_cell(0, 0, 0)]
    updated, reflex = evolve_step(grid, mock_config(), step=0)
    assert len(updated) >= 1
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex
    assert isinstance(updated[0], Cell)

def test_evolve_mixed_cells_with_solid_and_fluid():
    grid = [
        make_fluid_cell(0, 0, 0),
        make_solid_cell(1, 0, 0),
        make_fluid_cell(2, 0, 0)
    ]
    updated, reflex = evolve_step(grid, mock_config(), step=1)
    assert len(updated) >= 3
    for cell in updated:
        if getattr(cell, "fluid_mask", True):
            assert isinstance(cell.velocity, list)
            assert isinstance(cell.pressure, float)
        else:
            assert cell.velocity is None
            assert cell.pressure is None
    assert "max_velocity" in reflex

def test_evolve_empty_grid_does_not_crash():
    updated, reflex = evolve_step([], mock_config(), step=2)
    assert isinstance(updated, list)
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex

def test_malformed_velocity_downgrades_cell():
    bad_cell = Cell(x=0, y=0, z=0, velocity="bad", pressure=1.0, fluid_mask=True)
    grid = [bad_cell]
    updated, reflex = evolve_step(grid, mock_config(), step=3)
    downgraded = next(c for c in updated if c.x == 0 and c.y == 0 and c.z == 0)
    assert downgraded.fluid_mask is False
    assert downgraded.velocity is None
    assert downgraded.pressure is None
    assert isinstance(reflex["max_velocity"], float)

def test_reflex_metadata_has_expected_keys():
    grid = [make_fluid_cell(0, 0, 0)]
    _, reflex = evolve_step(grid, mock_config(), step=4)
    expected_keys = {
        "max_velocity",
        "global_cfl",
        "max_divergence",
        "damping_enabled",
        "overflow_detected",
        "adjusted_time_step",
        "projection_passes",
        "ghost_diagnostics"
    }
    for key in expected_keys:
        assert key in reflex

def test_consistent_step_metadata_across_steps():
    grid = [make_fluid_cell(0, 0, 0)]
    _, reflex_a = evolve_step(grid, mock_config(), step=0)
    _, reflex_b = evolve_step(grid, mock_config(), step=1)
    assert set(reflex_a.keys()) == set(reflex_b.keys())

def test_ghost_cells_in_diagnostics():
    grid = [make_fluid_cell(0, 0, 0)]
    _, reflex = evolve_step(grid, mock_config(), step=5)
    ghost_data = reflex.get("ghost_diagnostics", {})
    assert "total" in ghost_data
    assert isinstance(ghost_data["total"], int)
    assert "per_face" in ghost_data
    assert isinstance(ghost_data["per_face"], dict)
    assert sum(ghost_data["per_face"].values()) == ghost_data["total"]



