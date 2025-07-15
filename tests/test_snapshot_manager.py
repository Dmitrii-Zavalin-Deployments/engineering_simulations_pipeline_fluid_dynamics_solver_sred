# tests/test_snapshot_manager.py
# 🧪 Validates snapshot generation logic, reflex metadata handling, ghost injection, and output filtering

import pytest
import types
import tempfile
import shutil
import os
from src.snapshot_manager import generate_snapshots

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
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 0.2,
            "output_interval": 1
        },
        "boundary_conditions": {
            "apply_to": ["velocity", "pressure"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 99.0,
            "no_slip": True
        },
        "pressure_solver": {
            "method": "jacobi",
            "max_iterations": 10,
            "tolerance": 1e-5
        }
    }

@pytest.fixture
def config():
    return {
        "reflex_verbosity": "low",
        "include_pressure_mutation_map": False,
        "include_divergence_delta": False
    }

@pytest.fixture
def monkeypatch_evolve(monkeypatch):
    from src import step_controller
    def mock_evolve_step(grid, input_data, step, config=None):
        class DummyCell:
            def __init__(self, x): self.x = x; self.y = 0.0; self.z = 0.0
            velocity = [1.0, 0.0, 0.0]
            pressure = 5.0
            fluid_mask = True
        grid_out = [DummyCell(x) for x in [0.0, 1.0]]
        reflex = {
            "pressure_mutated": (step % 2 == 0),
            "velocity_projected": True,
            "projection_skipped": False,
            "mutated_cells": grid_out,
            "ghost_influence_count": 0,
            "pressure_solver_invoked": True,
            "max_divergence": 0.01,
            "ghost_registry": {},
            "fluid_cells_adjacent_to_ghosts": 0
        }
        return grid_out, reflex

    monkeypatch.setattr(step_controller, "evolve_step", mock_evolve_step)

def test_generate_snapshots_returns_correct_steps(input_data, config, monkeypatch_evolve):
    snapshots = generate_snapshots(input_data, scenario_name="test_case", config=config)
    assert isinstance(snapshots, list)
    assert all(isinstance(s, tuple) and isinstance(s[0], int) for s in snapshots)
    assert len(snapshots) == 3  # total_time / time_step = 0.2 / 0.1 → 3 steps (0,1,2)
    for step, snap in snapshots:
        assert snap["step_index"] == step
        assert "pressure_mutated" in snap
        assert "grid" in snap
        assert isinstance(snap["grid"], list)
        assert "ghost_diagnostics" in snap

def test_snapshot_contains_all_expected_reflex_fields(input_data, config, monkeypatch_evolve):
    snapshots = generate_snapshots(input_data, scenario_name="test_case", config=config)
    for _, snap in snapshots:
        keys = [
            "pressure_mutated", "velocity_projected",
            "projection_skipped", "ghost_diagnostics",
            "step_index", "grid"
        ]
        for k in keys:
            assert k in snap

def test_output_interval_filters_snapshots(monkeypatch_evolve):
    test_input = {
        "domain_definition": {"min_x":0,"max_x":1,"nx":1,"min_y":0,"max_y":1,"ny":1,"min_z":0,"max_z":1,"nz":1},
        "fluid_properties": {},
        "initial_conditions": {"velocity":[0,0,0],"pressure":0},
        "simulation_parameters": {"time_step":0.1,"total_time":0.5,"output_interval":2},
        "boundary_conditions": {},
        "pressure_solver": {}
    }
    config = {"reflex_verbosity":"low"}
    snapshots = generate_snapshots(test_input, scenario_name="case_interval", config=config)
    step_indices = [s[0] for s in snapshots]
    assert step_indices == [0, 2, 4]  # step % 2 == 0

def test_mutation_report_prints_summary(monkeypatch_evolve, capsys):
    test_input = {
        "domain_definition": {"min_x":0,"max_x":1,"nx":1,"min_y":0,"max_y":1,"ny":1,"min_z":0,"max_z":1,"nz":1},
        "fluid_properties": {},
        "initial_conditions": {"velocity":[0,0,0],"pressure":0},
        "simulation_parameters": {"time_step":0.1,"total_time":0.2,"output_interval":1},
        "boundary_conditions": {},
        "pressure_solver": {}
    }
    config = {"reflex_verbosity":"low"}
    snapshots = generate_snapshots(test_input, scenario_name="summary_case", config=config)
    out = capsys.readouterr().out
    assert "Pressure mutated steps" in out
    assert "Velocity projected steps" in out
    assert "Projection skipped steps" in out