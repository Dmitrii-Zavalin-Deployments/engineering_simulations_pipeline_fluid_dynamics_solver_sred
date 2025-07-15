# tests/snapshot/test_snapshot_manager.py
# 🧪 Unit tests for snapshot_manager.py — validates step evolution and snapshot composition

import os
import shutil
import tempfile
import pytest
from src.snapshot_manager import generate_snapshots
from src.grid_modules.cell import Cell

@pytest.fixture
def minimal_input_data():
    return {
        "simulation_parameters": {
            "time_step": 1.0,
            "total_time": 2.0,
            "output_interval": 1
        },
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 1, "nz": 1
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0
        }
    }

@pytest.fixture
def fake_config():
    return {
        "reflex_verbosity": "low",
        "include_divergence_delta": True,
        "include_pressure_mutation_map": True,
        "ghost_adjacency_depth": 1
    }

def test_snapshots_return_expected_steps(minimal_input_data, fake_config):
    scenario_name = "unit_test_run"
    result = generate_snapshots(minimal_input_data, scenario_name, fake_config)
    assert isinstance(result, list)
    assert len(result) == 3  # Steps 0, 1, 2
    for step_index, snapshot in result:
        assert isinstance(snapshot, dict)
        assert snapshot["step_index"] == step_index
        assert "grid" in snapshot
        assert "pressure_mutated" in snapshot
        assert "velocity_projected" in snapshot

def test_snapshot_grid_structure(minimal_input_data, fake_config):
    result = generate_snapshots(minimal_input_data, "struct_check", fake_config)
    _, snapshot = result[0]
    grid = snapshot["grid"]
    assert isinstance(grid, list)
    for cell in grid:
        assert isinstance(cell, dict)
        assert all(k in cell for k in ("x", "y", "z", "fluid_mask", "pressure"))
        if cell["fluid_mask"]:
            assert "velocity" in cell
        else:
            assert cell["velocity"] is None

def test_snapshot_respects_output_interval(minimal_input_data, fake_config):
    minimal_input_data["simulation_parameters"]["output_interval"] = 2
    result = generate_snapshots(minimal_input_data, "interval_check", fake_config)
    assert len(result) == 2  # Should include step 0 and step 2 only

def test_snapshot_creates_summary_and_logs(minimal_input_data, fake_config):
    folder_path = "data/testing-input-output/navier_stokes_output"
    shutil.rmtree(folder_path, ignore_errors=True)
    generate_snapshots(minimal_input_data, "summary_log_check", fake_config)
    assert os.path.exists(os.path.join(folder_path, "step_summary.txt"))
    assert os.path.exists(os.path.join(folder_path, "influence_flags_log.json"))
    assert os.path.exists(os.path.join(folder_path, "mutation_pathways_log.json"))

def test_mutation_pathway_serialization_safe(minimal_input_data, fake_config, monkeypatch):
    def mock_evolve_step(grid, input_data, step, config):
        dummy_cell = Cell(x=1.0, y=0.0, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=10.0, fluid_mask=True)
        return [dummy_cell], {
            "pressure_mutated": dummy_cell,  # intentionally wrong type
            "velocity_projected": True,
            "projection_skipped": False,
            "triggered_by": ["boundary_override"],
            "mutated_cells": [dummy_cell],
            "ghost_registry": [],
            "ghost_influence_count": 0,
            "fluid_cells_adjacent_to_ghosts": 0,
            "max_divergence": 0.0,
            "pressure_solver_invoked": True
        }

    monkeypatch.setattr("src.snapshot_manager.evolve_step", mock_evolve_step)
    result = generate_snapshots(minimal_input_data, "mutation_serialization_check", fake_config)
    assert len(result) == 3
    _, snapshot = result[0]
    assert isinstance(snapshot["pressure_mutated"], bool)  # Coerced from Cell



