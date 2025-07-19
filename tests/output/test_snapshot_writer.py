# tests/output/test_snapshot_writer.py
# 🧪 Unit tests for src/output/snapshot_writer.py

import os
import json
from pathlib import Path
from src.grid_modules.cell import Cell
from src.output.snapshot_writer import export_influence_flags

def make_cell(x=0.0, y=0.0, z=0.0, fluid=True, influenced=True):
    cell = Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=fluid)
    cell.influenced_by_ghost = influenced
    return cell

def test_export_creates_log_for_influenced_cells(tmp_path):
    folder = tmp_path / "flags"
    cell = make_cell(1, 2, 3)
    export_influence_flags([cell], step_index=5, output_folder=str(folder))

    log_file = folder / "influence_flags_log.json"
    assert log_file.exists()
    data = json.loads(log_file.read_text())
    assert isinstance(data, list)
    assert data[-1]["step_index"] == 5
    assert data[-1]["influenced_cell_count"] == 1
    assert data[-1]["cells"][0]["x"] == 1

def test_export_respects_quiet_mode(tmp_path, capsys):
    folder = tmp_path / "flags"
    cell = make_cell()
    export_influence_flags([cell], step_index=6, output_folder=str(folder), config={"reflex_verbosity": "low"})

    captured = capsys.readouterr()
    assert "[DEBUG]" not in captured.out

def test_export_includes_details_on_high_verbosity(tmp_path):
    folder = tmp_path / "flags"
    cell = make_cell()
    cell._ghost_velocity_source = (4, 5, 6)
    cell._ghost_pressure_source = (7, 8, 9)

    export_influence_flags([cell], step_index=7, output_folder=str(folder), config={"reflex_verbosity": "high"})
    log = json.loads((folder / "influence_flags_log.json").read_text())
    entry = log[-1]["cells"][0]
    assert "ghost_velocity_source" in entry
    assert "ghost_pressure_source" in entry
    assert "mutation_types" in entry
    assert entry["mutation_types"] == ["velocity", "pressure"]
    assert entry["velocity"] == [1.0, 0.0, 0.0]
    assert entry["pressure"] == 1.0

def test_export_skips_non_fluid_or_uninfluenced_cells(tmp_path):
    folder = tmp_path / "flags"
    safe = make_cell(fluid=True, influenced=False)
    solid = make_cell(fluid=False, influenced=True)
    export_influence_flags([safe, solid], step_index=8, output_folder=str(folder))

    log = json.loads((folder / "influence_flags_log.json").read_text())
    assert log[-1]["influenced_cell_count"] == 0
    assert log[-1]["cells"] == []



