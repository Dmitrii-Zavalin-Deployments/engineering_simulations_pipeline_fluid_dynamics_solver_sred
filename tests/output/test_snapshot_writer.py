# tests/output/test_snapshot_writer.py
# 🧪 Unit tests for snapshot_writer.py — validates influence flag export and log structure

import os
import json
import tempfile
from src.grid_modules.cell import Cell
from src.output.snapshot_writer import export_influence_flags

def make_cell(x, y, z, fluid=True, velocity=None, pressure=None, tag=False, tag_pressure=False, tag_velocity=False):
    cell = Cell(
        x=x, y=y, z=z,
        velocity=velocity if velocity is not None else [1.0, 0.0, 0.0],
        pressure=pressure if pressure is not None else 100.0,
        fluid_mask=fluid
    )
    if tag:
        cell.influenced_by_ghost = True
    if tag_pressure:
        cell._ghost_pressure_source = (x + 0.1, y + 0.1, z + 0.1)
    if tag_velocity:
        cell._ghost_velocity_source = (x - 0.1, y - 0.1, z - 0.1)
    return cell

def test_log_file_created_with_single_cell():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        cell = make_cell(0.5, 0.5, 0.5, tag=True, tag_pressure=True)
        export_influence_flags([cell], step_index=0, output_folder=tmpdir, config=config)

        path = os.path.join(tmpdir, "influence_flags_log.json")
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["step_index"] == 0
        assert data[0]["influenced_cell_count"] == 1
        assert len(data[0]["cells"]) == 1
        assert "mutation_types" in data[0]["cells"][0]
        assert "pressure" in data[0]["cells"][0]["mutation_types"]

def test_multiple_steps_append_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        c1 = make_cell(0, 0, 0, tag=True)
        c2 = make_cell(1, 1, 1, tag=True)

        export_influence_flags([c1], step_index=1, output_folder=tmpdir, config=config)
        export_influence_flags([c2], step_index=2, output_folder=tmpdir, config=config)

        path = os.path.join(tmpdir, "influence_flags_log.json")
        with open(path) as f:
            log = json.load(f)

        assert len(log) == 2
        assert log[0]["step_index"] == 1
        assert log[1]["step_index"] == 2

def test_non_influenced_cells_are_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        c = make_cell(0, 0, 0, tag=False)
        export_influence_flags([c], step_index=3, output_folder=tmpdir, config=config)

        path = os.path.join(tmpdir, "influence_flags_log.json")
        with open(path) as f:
            log = json.load(f)

        assert log[0]["step_index"] == 3
        assert log[0]["influenced_cell_count"] == 0
        assert log[0]["cells"] == []

def test_non_fluid_cells_are_skipped_even_if_tagged():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        c = make_cell(1, 1, 1, fluid=False, tag=True)
        export_influence_flags([c], step_index=4, output_folder=tmpdir, config=config)

        path = os.path.join(tmpdir, "influence_flags_log.json")
        with open(path) as f:
            log = json.load(f)

        assert log[0]["influenced_cell_count"] == 0
        assert log[0]["cells"] == []

def test_multiple_cells_tagged_and_logged():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        c1 = make_cell(1, 0, 0, tag=True, tag_velocity=True)
        c2 = make_cell(0, 1, 0, tag=True, tag_pressure=True)
        c3 = make_cell(0, 0, 1, tag=False)

        export_influence_flags([c1, c2, c3], step_index=5, output_folder=tmpdir, config=config)

        path = os.path.join(tmpdir, "influence_flags_log.json")
        with open(path) as f:
            log = json.load(f)

        assert log[0]["influenced_cell_count"] == 2
        coords = {(c["x"], c["y"], c["z"]) for c in log[0]["cells"]}
        assert (1, 0, 0) in coords
        assert (0, 1, 0) in coords
        assert (0, 0, 1) not in coords

def test_invalid_existing_log_file_recovers_cleanly():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        path = os.path.join(tmpdir, "influence_flags_log.json")
        with open(path, "w") as f:
            f.write("not-a-json-structure")

        cell = make_cell(1.0, 1.0, 1.0, tag=True)
        export_influence_flags([cell], step_index=6, output_folder=tmpdir, config=config)

        with open(path) as f:
            contents = f.read()

        assert "step_index" in contents
        assert "influenced_cell_count" in contents

def test_zero_cells_logged_still_creates_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        export_influence_flags([], step_index=7, output_folder=tmpdir, config=config)

        path = os.path.join(tmpdir, "influence_flags_log.json")
        with open(path) as f:
            log = json.load(f)

        assert len(log) == 1
        assert log[0]["step_index"] == 7
        assert log[0]["influenced_cell_count"] == 0
        assert log[0]["cells"] == []

def test_mutation_type_details_are_complete():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reflex_verbosity": "high"}
        c = make_cell(1, 1, 1, tag=True, tag_pressure=True, tag_velocity=True)
        export_influence_flags([c], step_index=99, output_folder=tmpdir, config=config)

        path = os.path.join(tmpdir, "influence_flags_log.json")
        with open(path) as f:
            log = json.load(f)
        cell_entry = log[0]["cells"][0]
        assert set(cell_entry["mutation_types"]) == {"pressure", "velocity"}
        assert "ghost_pressure_source" in cell_entry
        assert "ghost_velocity_source" in cell_entry



