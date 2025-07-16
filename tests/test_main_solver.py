# tests/test_main_solver.py

import os
import sys
import json
import pytest
import shutil
from unittest.mock import patch
from src import main_solver

OUTPUT_DIR = "data/testing-input-output/navier_stokes_output"
INPUT_FILE = "tests/data/mock_input.json"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "step_summary.txt")

@pytest.fixture(autouse=True)
def setup_environment(tmp_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def test_run_solver_executes_and_creates_snapshots(monkeypatch):
    sample_input = {
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 0.2,
            "output_interval": 1
        },
        "domain_definition": {
            "nx": 2, "ny": 2, "nz": 1,
            "min_x": 0, "max_x": 1,
            "min_y": 0, "max_y": 1,
            "min_z": 0, "max_z": 1
        },
        "initial_conditions": {},
        "geometry_definition": None
    }

    monkeypatch.setattr(main_solver, "load_simulation_input", lambda path: sample_input)
    monkeypatch.setattr(main_solver, "compact_pressure_delta_map", lambda a, b: None)

    main_solver.run_solver(INPUT_FILE, reflex_score_min=0)

    files = os.listdir(OUTPUT_DIR)
    snapshot_files = [f for f in files if f.endswith(".json")]
    assert len(snapshot_files) > 0

def test_load_reflex_config_fallback_on_missing():
    config = main_solver.load_reflex_config("nonexistent.yaml")
    assert isinstance(config, dict)
    assert config["reflex_verbosity"] == "medium"

def test_compaction_trigger_only_if_score_above_threshold(monkeypatch):
    trigger_steps = []
    def mock_generate_snapshots(*args, **kwargs):
        return [
            (0, {"reflex_score": 2}),
            (1, {"reflex_score": 5}),
        ]

    def mock_compact(original, compacted):
        trigger_steps.append(original)

    monkeypatch.setattr(main_solver, "generate_snapshots", mock_generate_snapshots)
    monkeypatch.setattr(main_solver, "load_simulation_input", lambda path: {})
    monkeypatch.setattr(main_solver, "compact_pressure_delta_map", mock_compact)

    main_solver.run_solver(INPUT_FILE, reflex_score_min=4)
    assert len(trigger_steps) == 1
    assert "step_0001" in trigger_steps[0]
