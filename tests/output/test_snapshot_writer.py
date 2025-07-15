# tests/test_snapshot_writer.py
# 🧪 Unit tests for export_influence_flags — verifies ghost influence logging across reflex verbosity modes

import os
import json
import shutil
import pytest
from src.output.snapshot_writer import export_influence_flags
from src.grid_modules.cell import Cell

TEST_DIR = "data/testing-output/influence_flags_tests"

@pytest.fixture
def clean_output_dir():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield TEST_DIR
    shutil.rmtree(TEST_DIR)

def make_cell(x=0.0, y=0.0, z=0.0, influenced=True, fluid=True):
    cell = Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=fluid)
    setattr(cell, "influenced_by_ghost", influenced)
    return cell

def make_detailed_cell(x=0.0, y=0.0, z=0.0):
    cell = make_cell(x, y, z, influenced=True, fluid=True)
    setattr(cell, "_ghost_velocity_source", (x-1, y, z))
    setattr(cell, "_ghost_pressure_source", (x, y-1, z))
    return cell

def test_low_verbosity_suppress_output(clean_output_dir):
    grid = [make_cell(x=1.0, y=2.0, z=3.0)]
    config = {"reflex_verbosity": "low"}
    export_influence_flags(grid, step_index=0, output_folder=clean_output_dir, config=config)

    path = os.path.join(clean_output_dir, "influence_flags_log.json")
    with open(path) as f:
        log = json.load(f)
    assert log[0]["influenced_cell_count"] == 1
    assert "ghost_pressure_source" not in log[0]["cells"][0]

def test_high_verbosity_includes_mutation_fields(clean_output_dir):
    grid = [make_detailed_cell(x=5.0, y=5.0, z=5.0)]
    config = {"reflex_verbosity": "high"}
    export_influence_flags(grid, step_index=1, output_folder=clean_output_dir, config=config)

    path = os.path.join(clean_output_dir, "influence_flags_log.json")
    with open(path) as f:
        log = json.load(f)
    cell_entry = log[0]["cells"][0]
    assert "ghost_pressure_source" in cell_entry
    assert "ghost_velocity_source" in cell_entry
    assert cell_entry["mutation_types"] == ["velocity", "pressure"]
    assert isinstance(cell_entry["velocity"], list)
    assert isinstance(cell_entry["pressure"], (int, float))

def test_medium_verbosity_excludes_mutation_fields(clean_output_dir):
    grid = [make_detailed_cell()]
    export_influence_flags(grid, step_index=2, output_folder=clean_output_dir)
    path = os.path.join(clean_output_dir, "influence_flags_log.json")
    with open(path) as f:
        log = json.load(f)
    cell_entry = log[0]["cells"][0]
    assert "mutation_types" not in cell_entry
    assert "ghost_pressure_source" not in cell_entry
    assert "ghost_velocity_source" not in cell_entry

def test_solid_cell_skipped(clean_output_dir):
    grid = [
        make_cell(x=1.0, y=1.0, z=1.0, influenced=True, fluid=False),
        make_cell(x=2.0, y=2.0, z=2.0, influenced=True, fluid=True),
    ]
    export_influence_flags(grid, step_index=3, output_folder=clean_output_dir)
    path = os.path.join(clean_output_dir, "influence_flags_log.json")
    with open(path) as f:
        log = json.load(f)
    assert log[0]["influenced_cell_count"] == 1
    coords = [(c["x"], c["y"], c["z"]) for c in log[0]["cells"]]
    assert (2.0, 2.0, 2.0) in coords
    assert (1.0, 1.0, 1.0) not in coords

def test_uninfluenced_cell_not_logged(clean_output_dir):
    grid = [make_cell(influenced=False)]
    export_influence_flags(grid, step_index=4, output_folder=clean_output_dir)
    path = os.path.join(clean_output_dir, "influence_flags_log.json")
    with open(path) as f:
        log = json.load(f)
    assert log[0]["influenced_cell_count"] == 0
    assert log[0]["cells"] == []

def test_multiple_steps_appended(clean_output_dir):
    grid = [make_cell(influenced=True)]
    for i in range(3):
        export_influence_flags(grid, step_index=i, output_folder=clean_output_dir)

    path = os.path.join(clean_output_dir, "influence_flags_log.json")
    with open(path) as f:
        log = json.load(f)
    assert len(log) == 3
    steps = [entry["step_index"] for entry in log]
    assert steps == [0, 1, 2]

def test_existing_corrupt_log_recovered(clean_output_dir):
    log_path = os.path.join(clean_output_dir, "influence_flags_log.json")
    with open(log_path, "w") as f:
        f.write("corrupt-non-json")
    grid = [make_cell(influenced=True)]
    export_influence_flags(grid, step_index=7, output_folder=clean_output_dir)
    with open(log_path) as f:
        log = json.load(f)
    assert isinstance(log, list)
    assert log[0]["step_index"] == 7