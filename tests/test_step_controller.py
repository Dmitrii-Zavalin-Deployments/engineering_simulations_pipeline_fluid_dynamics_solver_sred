# tests/test_step_controller.py
# 🧪 Validates evolve_step orchestration, reflex field injection, ghost handling, divergence reporting

import pytest
from src.step_controller import evolve_step
from src.grid_modules.cell import Cell

@pytest.fixture
def input_data():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "fluid_properties": {"viscosity": 0.5},
        "initial_conditions": {"velocity": [1.0, 0.0, 0.0], "pressure": 5.0},
        "simulation_parameters": {"time_step": 0.1, "output_interval": 1},
        "boundary_conditions": {
            "apply_to": ["velocity", "pressure"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 99.0,
            "no_slip": True
        },
        "pressure_solver": {"method": "jacobi", "tolerance": 1e-5}
    }

@pytest.fixture
def dummy_grid():
    return [Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0], pressure=5.0, fluid_mask=True)]

def test_evolve_step_runs_successfully(dummy_grid, input_data):
    result_grid, reflex = evolve_step(dummy_grid, input_data, step=0)
    assert isinstance(result_grid, list)
    assert all(isinstance(cell, Cell) for cell in result_grid)
    assert isinstance(reflex, dict)
    assert "max_divergence" in reflex
    assert "pressure_mutated" in reflex
    assert "ghost_registry" in reflex
    assert "projection_passes" in reflex

def test_evolve_step_output_grid_contains_velocity_and_pressure(dummy_grid, input_data):
    result_grid, reflex = evolve_step(dummy_grid, input_data, step=1)
    fluid_cells = [c for c in result_grid if c.fluid_mask]
    assert all(isinstance(c.velocity, list) for c in fluid_cells)
    assert all(isinstance(c.pressure, float) for c in fluid_cells)

def test_evolve_step_injects_reflex_diagnostics(dummy_grid, input_data):
    result_grid, reflex = evolve_step(dummy_grid, input_data, step=2)
    assert "ghost_diagnostics" in reflex
    diag = reflex["ghost_diagnostics"]
    assert isinstance(diag, dict)
    assert "total" in diag
    assert "pressure_overrides" in diag

def test_evolve_step_merges_pressure_metadata(dummy_grid, input_data):
    result_grid, reflex = evolve_step(dummy_grid, input_data, step=3)
    assert "pressure_mutation_count" in reflex
    assert isinstance(reflex["pressure_mutation_count"], int)
    assert "mutated_cells" in reflex
    assert isinstance(reflex["mutated_cells"], list)

def test_evolve_step_handles_nonstandard_pressure_mutated(dummy_grid, input_data):
    # Patch apply_pressure_correction to return non-boolean pressure_mutated
    from src import solvers
    def patched_pressure_solver(grid, data, step):
        return grid, {"unexpected": "object"}, 1, {"mutated_cells": []}
    original = solvers.pressure_solver.apply_pressure_correction
    solvers.pressure_solver.apply_pressure_correction = patched_pressure_solver
    try:
        result_grid, reflex = evolve_step(dummy_grid, input_data, step=4)
        assert isinstance(reflex["pressure_mutated"], bool)
    finally:
        solvers.pressure_solver.apply_pressure_correction = original

def test_evolve_step_divergence_stats_logged(tmp_path, dummy_grid, input_data):
    input_data["simulation_parameters"]["output_interval"] = 1
    folder = "data/testing-input-output/navier_stokes_output"
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    evolve_step(dummy_grid, input_data, step=5)
    log_path = os.path.join(folder, "divergence_log.txt")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        lines = f.readlines()
    assert any("before projection" in line for line in lines)
    assert any("after projection" in line for line in lines)