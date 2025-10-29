# tests/output/test_snapshot_writer.py
# ✅ Validation suite for src/output/snapshot_writer.py

import pytest
import os
import json
import tempfile
from src.output.snapshot_writer import export_influence_flags
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True, influenced=False):
    cell = Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )
    cell.influenced_by_ghost = influenced
    return cell

def test_export_influence_flags_creates_log_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        cell = make_cell(1.0, 2.0, 3.0, influenced=True)
        export_influence_flags([cell], step_index=0, output_folder=temp_dir)

        log_path = os.path.join(temp_dir, "influence_flags_log.json")
        assert os.path.isfile(log_path)

        with open(log_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["step_index"] == 0
        assert entry["influenced_cell_count"] == 1
        assert entry["cells"][0]["x"] == 1.0
        assert entry["cells"][0]["y"] == 2.0
        assert entry["cells"][0]["z"] == 3.0

def test_export_influence_flags_respects_fluid_mask():
    with tempfile.TemporaryDirectory() as temp_dir:
        cell = make_cell(0.0, 0.0, 0.0, fluid_mask=False, influenced=True)
        export_influence_flags([cell], step_index=1, output_folder=temp_dir)

        log_path = os.path.join(temp_dir, "influence_flags_log.json")
        with open(log_path, "r") as f:
            data = json.load(f)

        assert data[0]["influenced_cell_count"] == 0
        assert data[0]["cells"] == []

def test_export_influence_flags_appends_multiple_steps():
    with tempfile.TemporaryDirectory() as temp_dir:
        cell1 = make_cell(0.0, 0.0, 0.0, influenced=True)
        cell2 = make_cell(1.0, 1.0, 1.0, influenced=True)

        export_influence_flags([cell1], step_index=0, output_folder=temp_dir)
        export_influence_flags([cell2], step_index=1, output_folder=temp_dir)

        log_path = os.path.join(temp_dir, "influence_flags_log.json")
        with open(log_path, "r") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["step_index"] == 0
        assert data[1]["step_index"] == 1
        assert data[0]["influenced_cell_count"] == 1
        assert data[1]["influenced_cell_count"] == 1

def test_export_influence_flags_includes_details_when_high_verbosity():
    with tempfile.TemporaryDirectory() as temp_dir:
        cell = make_cell(2.0, 2.0, 2.0, influenced=True)
        cell._ghost_velocity_source = (0.0, 0.0, 0.0)
        cell._ghost_pressure_source = (1.0, 1.0, 1.0)

        config = {"reflex_verbosity": "high"}
        export_influence_flags([cell], step_index=2, output_folder=temp_dir, config=config)

        log_path = os.path.join(temp_dir, "influence_flags_log.json")
        with open(log_path, "r") as f:
            data = json.load(f)

        entry = data[0]["cells"][0]
        assert entry["mutation_types"] == ["velocity", "pressure"]
        assert entry["ghost_velocity_source"] == [0.0, 0.0, 0.0]
        assert entry["ghost_pressure_source"] == [1.0, 1.0, 1.0]
        assert "velocity" in entry
        assert "pressure" in entry

def test_export_influence_flags_suppresses_output_when_verbosity_low():
    with tempfile.TemporaryDirectory() as temp_dir:
        cell = make_cell(3.0, 3.0, 3.0, influenced=True)
        config = {"reflex_verbosity": "low"}
        export_influence_flags([cell], step_index=3, output_folder=temp_dir, config=config)

        log_path = os.path.join(temp_dir, "influence_flags_log.json")
        with open(log_path, "r") as f:
            data = json.load(f)

        assert "mutation_types" not in data[0]["cells"][0]
        assert "ghost_velocity_source" not in data[0]["cells"][0]
        assert "ghost_pressure_source" not in data[0]["cells"][0]
        assert "velocity" not in data[0]["cells"][0]
        assert "pressure" not in data[0]["cells"][0]



