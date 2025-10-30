# tests/test_snapshot_manager.py

import os
import pytest
from unittest import mock
import src.snapshot_manager
from src.snapshot_manager import generate_snapshots

@pytest.fixture
def input_data():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 1
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0.0, 0.0, 0.0], "initial_pressure": 101325.0},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.03, "output_interval": 1},
        "geometry_definition": {
            "geometry_mask_flat": [1, 1, 1, 1],
            "geometry_mask_shape": [2, 2, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }

@pytest.fixture
def reflex_config():
    return {
        "reflex_verbosity": "high",
        "include_divergence_delta": True,
        "include_pressure_mutation_map": True,
        "log_projection_trace": True,
        "ghost_adjacency_depth": 2
    }

@pytest.fixture
def sim_config():
    return {}

@pytest.fixture
def output_dir(tmp_path):
    path = tmp_path / "navier_stokes_output"
    path.mkdir(parents=True, exist_ok=True)
    return path

def test_generate_snapshots_runs_all_steps(monkeypatch, input_data, reflex_config, sim_config, output_dir):
    monkeypatch.setitem(
        src.snapshot_manager.generate_snapshots.__globals__,
        "generate_grid_with_mask",
        lambda d, i, g: [mock.Mock(fluid_mask=True) for _ in range(4)]
    )

    def mock_evolve_step(grid, input_data, step, config=None, sim_config=None):
        return grid, {"reflex_score": 4.0}
    monkeypatch.setattr("src.snapshot_manager.evolve_step", mock_evolve_step)

    def mock_write_velocity_field(grid, step, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"velocity_field_step_{step:04d}.json"), "w") as f:
            f.write("{}")
    monkeypatch.setattr("src.snapshot_manager.write_velocity_field", mock_write_velocity_field)

    def mock_process_snapshot_step(step, grid, reflex, spacing, config, expected_size, output_folder, sim_config=None):
        return grid, {
            "reflex_score": 4.0,
            "pressure_mutated": step == 1,
            "velocity_projected": True,
            "projection_skipped": step == 2
        }
    monkeypatch.setattr("src.snapshot_manager.process_snapshot_step", mock_process_snapshot_step)

    snapshots = generate_snapshots(sim_config, "test_scenario", reflex_config, output_dir=str(output_dir))
    assert len(snapshots) == 4
    assert snapshots[0][0] == 0
    assert snapshots[-1][0] == 3
    for _, snap in snapshots:
        for key in ["reflex_score", "pressure_mutated", "velocity_projected", "projection_skipped"]:
            assert key in snap

def test_generate_snapshots_tracks_mutations(monkeypatch, input_data, reflex_config, sim_config, output_dir):
    monkeypatch.setitem(
        src.snapshot_manager.generate_snapshots.__globals__,
        "generate_grid_with_mask",
        lambda d, i, g: [mock.Mock(fluid_mask=True) for _ in range(4)]
    )

    def mock_write_velocity_field(grid, step, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"velocity_field_step_{step:04d}.json"), "w") as f:
            f.write("{}")
    monkeypatch.setattr("src.snapshot_manager.write_velocity_field", mock_write_velocity_field)

    monkeypatch.setattr("src.snapshot_manager.evolve_step", lambda g, i, s, config=None, sim_config=None: (g, {}))

    mutation_map = {
        0: {"pressure_mutated": False, "velocity_projected": True, "projection_skipped": False},
        1: {"pressure_mutated": True, "velocity_projected": True, "projection_skipped": False},
        2: {"pressure_mutated": False, "velocity_projected": True, "projection_skipped": True},
        3: {"pressure_mutated": False, "velocity_projected": True, "projection_skipped": False}
    }

    def mock_process(step, grid, reflex, spacing, config, expected_size, output_folder, sim_config=None):
        return grid, {"reflex_score": 3.5, **mutation_map[step]}
    monkeypatch.setattr("src.snapshot_manager.process_snapshot_step", mock_process)

    snapshots = generate_snapshots(sim_config, "mutation_test", reflex_config, output_dir=str(output_dir))
    scores = [snap[1]["reflex_score"] for snap in snapshots]
    assert all(isinstance(s, float) for s in scores)
    assert sum(snap[1]["pressure_mutated"] for snap in snapshots) == 1
    assert sum(snap[1]["velocity_projected"] for snap in snapshots) == 4
    assert sum(snap[1]["projection_skipped"] for snap in snapshots) == 1

def test_generate_snapshots_raises_on_invalid_output_interval(monkeypatch, input_data, reflex_config, sim_config, output_dir):
    input_data["simulation_parameters"]["output_interval"] = 0

    monkeypatch.setitem(
        src.snapshot_manager.generate_snapshots.__globals__,
        "generate_grid_with_mask",
        lambda d, i, g: [mock.Mock(fluid_mask=True) for _ in range(4)]
    )

    monkeypatch.setattr("src.snapshot_manager.write_velocity_field", lambda *a, **kw: None)
    monkeypatch.setattr("src.snapshot_manager.evolve_step", lambda *a, **kw: ([], {}))
    monkeypatch.setattr("src.snapshot_manager.process_snapshot_step", lambda *a, **kw: ([], {
        "reflex_score": 1.0,
        "pressure_mutated": False,
        "velocity_projected": True,
        "projection_skipped": False
    }))

    with pytest.raises(ValueError, match="Invalid output_interval: 0"):
        generate_snapshots(sim_config, "fallback_test", reflex_config, output_dir=str(output_dir))
