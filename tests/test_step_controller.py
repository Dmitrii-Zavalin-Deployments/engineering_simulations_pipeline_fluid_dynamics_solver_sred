# tests/test_step_controller.py
# 🧪 Tests for simulation step controller — verifies grid evolution, ghost diagnostics, influence, and reflex metadata

import pytest
from src.step_controller import evolve_step
from src.grid_modules.cell import Cell

def make_fluid_cell(x, y, z, velocity=None, pressure=10.0):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity if velocity is not None else [0.0, 0.0, 0.0],
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
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "boundary_conditions": {
            "apply_faces": ["x_min", "x_max"],
            "pressure": 20.0,
            "velocity": [0.4, 0.0, 0.0],
            "no_slip": True
        }
    }

def test_single_fluid_cell_evolves():
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    updated, reflex = evolve_step(grid, mock_config(), step=0)
    assert isinstance(updated, list)
    assert isinstance(reflex, dict)
    assert any(isinstance(c, Cell) for c in updated)
    assert "max_velocity" in reflex

def test_ghost_diagnostics_have_valid_fields():
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    _, reflex = evolve_step(grid, mock_config(), step=1)
    ghost = reflex.get("ghost_diagnostics", {})
    assert isinstance(ghost.get("total", None), int)
    assert isinstance(ghost.get("per_face", None), dict)
    assert "fluid_cells_adjacent_to_ghosts" in ghost

def test_ghost_influence_modifies_fluid():
    fluid = make_fluid_cell(1.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    grid = [fluid]
    updated, reflex = evolve_step(grid, mock_config(), step=2)
    modified = next(c for c in updated if c.x == fluid.x and c.y == fluid.y and c.z == fluid.z)
    assert modified.velocity != [0.0, 0.0, 0.0]
    assert modified.pressure != 0.0

def test_empty_grid_has_safe_reflex():
    updated, reflex = evolve_step([], mock_config(), step=3)
    assert isinstance(updated, list)
    assert isinstance(reflex, dict)
    assert "max_velocity" in reflex
    assert reflex.get("projection_passes", 0) >= 0

def test_malformed_velocity_downgrades_cell():
    bad = Cell(x=1.0, y=0.5, z=0.5, velocity="corrupt", pressure=10.0, fluid_mask=True)
    updated, reflex = evolve_step([bad], mock_config(), step=4)
    downgraded = next(c for c in updated if c.x == 1.0)
    assert downgraded.fluid_mask is False
    assert downgraded.velocity is None
    assert downgraded.pressure is None

def test_divergence_tracking_runs_pre_and_post():
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    # Capture console output from divergence tracker
    import io, sys
    buffer = io.StringIO()
    sys.stdout = buffer
    evolve_step(grid, mock_config(), step=5)
    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
    assert "Divergence stats (before projection)" in output
    assert "Divergence stats (after projection)" in output

def test_step_consistency_across_iterations():
    grid = [make_fluid_cell(1.0, 0.5, 0.5)]
    _, reflex_a = evolve_step(grid, mock_config(), step=6)
    _, reflex_b = evolve_step(grid, mock_config(), step=7)
    assert set(reflex_a.keys()) == set(reflex_b.keys())
    assert "ghost_diagnostics" in reflex_a

def test_boundary_conditions_enforced_via_ghosts():
    grid = [make_fluid_cell(1.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=None)]
    updated, reflex = evolve_step(grid, mock_config(), step=8)
    cell = next(c for c in updated if getattr(c, "fluid_mask", True))
    assert cell.pressure is not None
    assert cell.velocity != [0.0, 0.0, 0.0]

def test_multiple_fluid_cells_influence():
    grid = [
        make_fluid_cell(1.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=0.0),
        make_fluid_cell(2.0, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    ]
    updated, reflex = evolve_step(grid, mock_config(), step=9)
    influenced = [c for c in updated if getattr(c, "fluid_mask", False) and c.pressure != 0.0]
    assert len(influenced) == 2
    assert all(c.velocity != [0.0, 0.0, 0.0] for c in influenced)



