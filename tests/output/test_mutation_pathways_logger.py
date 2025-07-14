# tests/output/test_mutation_pathways_logger.py
# 🧪 Unit tests for mutation_pathways_logger.py — validates pressure mutation trace export

import os
import json
import tempfile
import pytest
from src.grid_modules.cell import Cell
from src.output.mutation_pathways_logger import log_mutation_pathway

def test_log_file_created_and_contains_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_mutation_pathway(step_index=0, pressure_mutated=True, triggered_by=["ghost_influence"], output_folder=tmpdir)
        path = os.path.join(tmpdir, "mutation_pathways_log.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["step_index"] == 0
        assert entry["pressure_mutated"] is True
        assert entry["triggered_by"] == ["ghost_influence"]

def test_multiple_entries_append_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_mutation_pathway(1, False, ["none"], tmpdir)
        log_mutation_pathway(2, True, ["boundary_override"], tmpdir)
        path = os.path.join(tmpdir, "mutation_pathways_log.json")
        with open(path) as f:
            log = json.load(f)
        assert len(log) == 2
        assert log[0]["step_index"] == 1
        assert log[1]["step_index"] == 2

def test_empty_triggered_by_list_is_allowed():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_mutation_pathway(3, True, [], tmpdir)
        path = os.path.join(tmpdir, "mutation_pathways_log.json")
        with open(path) as f:
            log = json.load(f)
        assert log[-1]["triggered_by"] == []

def test_invalid_existing_file_is_overwritten_safely():
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "mutation_pathways_log.json")
        with open(corrupt_path, "w") as f:
            f.write("corrupt[json")
        log_mutation_pathway(4, False, ["ghost_influence"], tmpdir)
        with open(corrupt_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["step_index"] == 4

def test_non_list_existing_content_is_reset():
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = os.path.join(tmpdir, "mutation_pathways_log.json")
        with open(invalid_path, "w") as f:
            json.dump({"not": "a list"}, f)
        log_mutation_pathway(5, True, ["ghost_influence"], tmpdir)
        with open(invalid_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["step_index"] == 5

def test_custom_output_folder_path_works():
    with tempfile.TemporaryDirectory() as base:
        custom_folder = os.path.join(base, "nested/output/results")
        log_mutation_pathway(6, True, ["ghost_influence", "boundary_override"], output_folder=custom_folder)
        log_path = os.path.join(custom_folder, "mutation_pathways_log.json")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert entries[0]["triggered_by"] == ["ghost_influence", "boundary_override"]

def test_triggered_cells_serialization_accepts_cell_objects():
    with tempfile.TemporaryDirectory() as tmpdir:
        cells = [
            Cell(x=0.0, y=1.0, z=2.0, velocity=[1.0, 0.0, 0.0], pressure=100.0, fluid_mask=True),
            Cell(x=1.0, y=0.0, z=2.0, velocity=None, pressure=None, fluid_mask=False)
        ]
        log_mutation_pathway(7, True, ["test"], tmpdir, triggered_cells=cells)
        path = os.path.join(tmpdir, "mutation_pathways_log.json")
        with open(path) as f:
            log = json.load(f)
        entry = log[-1]
        assert "triggered_cells" in entry
        assert isinstance(entry["triggered_cells"], list)
        assert entry["triggered_cells"][0]["x"] == 0.0
        assert entry["triggered_cells"][1]["fluid_mask"] is False

def test_triggered_cells_serialization_accepts_tuples():
    with tempfile.TemporaryDirectory() as tmpdir:
        coords = [(0.0, 1.0, 2.0), (3.0, 3.0, 3.0)]
        log_mutation_pathway(8, False, ["tuple_test"], tmpdir, triggered_cells=coords)
        path = os.path.join(tmpdir, "mutation_pathways_log.json")
        with open(path) as f:
            log = json.load(f)
        entry = log[-1]
        assert "triggered_cells" in entry
        assert entry["triggered_cells"][1]["x"] == 3.0
        assert entry["triggered_cells"][1]["y"] == 3.0
        assert entry["triggered_cells"][1]["z"] == 3.0



