# tests/test_step_controller.py
# 🧪 Tests for simulation step controller — verifies grid evolution, ghost diagnostics, influence, and reflex metadata

import os
import io
import sys
import pytest
from src.step_controller import evolve_step
from src.grid_modules.cell import Cell
from src.main_solver import load_reflex_config

def make_fluid_cell(x, y, z, velocity=None, pressure=10.0):
    return Cell(
        x=x, y=y, z=z,
        velocity=velocity if velocity is not None else [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=True
    )

def make_solid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)

def mock_input_config(time_step=0.1, viscosity=0.01):
    return {
        "simulation_parameters": { "time_step": time_step },
        "fluid_properties": { "viscosity": viscosity },
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "boundary_conditions": {
            "apply_faces": ["x_min", "x_max"],
            "pressure": 20.0,
            "velocity": [0.4, 0.0, 0.0],
            "no_slip": True,
            "apply_to": ["pressure", "velocity"]
        }
    }

@pytest.fixture
def reflex_config():
    return load_reflex_config("config/reflex_debug_config.yaml")

def test_single_fluid_cell_evolves(reflex_config):
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    updated, reflex = evolve_step(grid, mock_input_config(), step=0, config=reflex_config)
    assert isinstance(updated, list)
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex
    assert "boundary_condition_applied" in reflex
    assert "ghost_influence_count" in reflex

def test_ghost_registry_and_influence_logged(reflex_config):
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    _, reflex = evolve_step(grid, mock_input_config(), step=1, config=reflex_config)
    assert "ghost_registry" in reflex
    assert isinstance(reflex["ghost_influence_count"], int)

def test_ghost_diagnostics_have_valid_fields(reflex_config):
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    _, reflex = evolve_step(grid, mock_input_config(), step=2, config=reflex_config)
    ghost = reflex.get("ghost_diagnostics", {})
    assert isinstance(ghost.get("total", None), int)
    assert isinstance(ghost.get("per_face", None), dict)
    assert "fluid_cells_adjacent_to_ghosts" in ghost

def test_ghost_influence_modifies_fluid_and_tags(reflex_config):
    fluid = make_fluid_cell(1.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    grid = [fluid]
    updated, reflex = evolve_step(grid, mock_input_config(), step=3, config=reflex_config)
    modified = next(c for c in updated if c.x == fluid.x and c.y == fluid.y and c.z == fluid.z)
    assert modified.velocity != [0.0, 0.0, 0.0]
    assert modified.pressure != 0.0
    assert getattr(modified, "influenced_by_ghost", False)
    assert reflex["ghost_influence_count"] >= 1

def test_empty_grid_has_safe_reflex(reflex_config):
    updated, reflex = evolve_step([], mock_input_config(), step=4, config=reflex_config)
    assert isinstance(updated, list)
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex
    assert reflex.get("projection_passes", 0) >= 0

def test_malformed_velocity_downgrades_cell(reflex_config):
    bad = Cell(x=1.0, y=0.5, z=0.5, velocity="corrupt", pressure=10.0, fluid_mask=True)
    updated, reflex = evolve_step([bad], mock_input_config(), step=5, config=reflex_config)
    downgraded = next(c for c in updated if c.x == 1.0)
    assert downgraded.fluid_mask is False
    assert downgraded.velocity is None
    assert downgraded.pressure is None

def test_divergence_tracking_runs_and_writes_log(reflex_config):
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    log_path = "data/testing-input-output/navier_stokes_output/divergence_log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)
    buffer = io.StringIO()
    sys.stdout = buffer
    evolve_step(grid, mock_input_config(), step=6, config=reflex_config)
    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
    assert "Divergence stats (before projection)" in output
    assert "Divergence stats (after projection)" in output
    assert os.path.exists(log_path)
    with open(log_path) as f:
        contents = f.read()
    assert "Step 0006" in contents

def test_step_consistency_across_iterations(reflex_config):
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    _, reflex_a = evolve_step(grid, mock_input_config(), step=7, config=reflex_config)
    _, reflex_b = evolve_step(grid, mock_input_config(), step=8, config=reflex_config)
    assert set(reflex_a.keys()) == set(reflex_b.keys())

def test_boundary_conditions_enforced_via_ghosts(reflex_config):
    grid = [make_fluid_cell(1.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=None)]
    updated, reflex = evolve_step(grid, mock_input_config(), step=9, config=reflex_config)
    cell = next(c for c in updated if getattr(c, "fluid_mask", True))
    assert cell.pressure is not None
    assert cell.velocity != [0.0, 0.0, 0.0]
    assert reflex.get("boundary_condition_applied", False)

def test_multiple_fluid_cells_influence_and_tags(reflex_config):
    grid = [
        make_fluid_cell(1.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=0.0),
        make_fluid_cell(2.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    ]
    updated, reflex = evolve_step(grid, mock_input_config(), step=10, config=reflex_config)
    influenced = [c for c in updated if getattr(c, "fluid_mask", False) and getattr(c, "influenced_by_ghost", False)]
    assert len(influenced) == 2
    assert all(c.velocity != [0.0, 0.0, 0.0] for c in influenced)
    assert reflex["ghost_influence_count"] == 2



