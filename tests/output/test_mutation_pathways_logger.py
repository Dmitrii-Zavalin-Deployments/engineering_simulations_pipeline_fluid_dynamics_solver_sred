# tests/output/test_mutation_pathways_logger.py
# ✅ Validation suite for src/output/mutation_pathways_logger.py

import pytest
import os
import json
import tempfile
from src.output.mutation_pathways_logger import serialize_cell, log_mutation_pathway
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )

def test_serialize_cell_from_cell_object():
    cell = make_cell(1.0, 2.0, 3.0)
    cell.mutation_triggered_by = ["ghost_influence"]
    cell.pressure_delta = 0.5
    cell.influenced_by_ghost = True
    cell.mutation_source = "velocity_overflow"
    cell.mutation_step = 42

    result = serialize_cell(cell, reason="boundary", step_linked_from=41)
    assert result["x"] == 1.0
    assert result["y"] == 2.0
    assert result["z"] == 3.0
    assert result["pressure_changed"] is False
    assert result["suppression_reason"] == "boundary"
    assert result["step_linked_from"] == 41
    assert result["triggered_by"] == ["ghost_influence"]
    assert result["pressure_delta"] == 0.5
    assert result["influenced_by_ghost"] is True
    assert result["mutation_source"] == "velocity_overflow"
    assert result["mutation_step"] == 42

def test_serialize_cell_from_tuple():
    result = serialize_cell((1.0, 2.0, 3.0), reason="ghost", step_linked_from=99)
    assert result["x"] == 1.0
    assert result["y"] == 2.0
    assert result["z"] == 3.0
    assert result["pressure_changed"] is True
    assert result["suppression_reason"] == "ghost"
    assert result["step_linked_from"] == 99

def test_log_mutation_pathway_creates_and_appends_log():
    with tempfile.TemporaryDirectory() as temp_dir:
        cell1 = make_cell(0.0, 0.0, 0.0)
        cell2 = make_cell(1.0, 1.0, 1.0)
        cell2.mutation_triggered_by = ["ghost_influence"]
        cell2.pressure_delta = 0.2
        cell2.influenced_by_ghost = True
        cell2.mutation_source = "velocity_overflow"
        cell2.mutation_step = 5

        log_mutation_pathway(
            step_index=5,
            pressure_mutated=True,
            triggered_by=["ghost_influence"],
            output_folder=temp_dir,
            triggered_cells=[cell1, cell2],
            ghost_trigger_chain=[3, 4]
        )

        log_path = os.path.join(temp_dir, "mutation_pathways_log.json")
        assert os.path.isfile(log_path)

        with open(log_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["step_index"] == 5
        assert entry["pressure_mutated"] is True
        assert entry["triggered_by"] == ["ghost_influence"]
        assert entry["ghost_trigger_chain"] == [3, 4]
        assert "triggered_cells" in entry
        assert "mutated_cells" in entry
        assert len(entry["mutated_cells"]) == 2
        assert entry["mutated_cells"][0] == [0.0, 0.0, 0.0]  # ✅ Fixed: tuple → list

def test_log_mutation_pathway_handles_missing_log_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(temp_dir, "mutation_pathways_log.json")
        assert not os.path.exists(log_path)

        log_mutation_pathway(
            step_index=1,
            pressure_mutated=False,
            triggered_by=[],
            output_folder=temp_dir,
            triggered_cells=None,
            ghost_trigger_chain=None
        )

        assert os.path.isfile(log_path)
        with open(log_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["step_index"] == 1
        assert data[0]["pressure_mutated"] is False
        assert data[0]["triggered_by"] == []
        assert data[0]["ghost_trigger_chain"] == []
        assert "triggered_cells" not in data[0]
        assert "mutated_cells" not in data[0]



