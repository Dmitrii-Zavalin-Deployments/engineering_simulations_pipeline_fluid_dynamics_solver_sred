# tests/output/test_mutation_pathways_logger.py
# 🧪 Unit tests for src/output/mutation_pathways_logger.py

import os
import json
import tempfile
from src.grid_modules.cell import Cell
from src.output.mutation_pathways_logger import (
    log_mutation_pathway,
    log_skipped_mutation,
    serialize_cell
)

def make_cell(x=0.0, y=0.0, z=0.0):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)

def test_log_mutation_pathway_creates_expected_entry(tmp_path):
    folder = tmp_path / "output"
    os.makedirs(folder, exist_ok=True)

    cells = [make_cell(1, 2, 3), make_cell(4, 5, 6)]
    log_mutation_pathway(
        step_index=10,
        pressure_mutated=True,
        triggered_by=["ghost trigger"],
        output_folder=str(folder),
        triggered_cells=cells
    )

    log_file = folder / "mutation_pathways_log.json"
    assert log_file.exists()

    with open(log_file) as f:
        data = json.load(f)

    entry = data[-1]
    assert entry["step_index"] == 10
    assert entry["pressure_mutated"] is True
    assert entry["triggered_by"] == ["ghost trigger"]
    assert "triggered_cells" in entry
    assert "mutated_cells" in entry
    assert len(entry["mutated_cells"]) == 2
    assert entry["mutated_cells"][0] == (1, 2, 3)

def test_log_skipped_mutation_creates_expected_entry(tmp_path):
    folder = tmp_path / "output"
    os.makedirs(folder, exist_ok=True)

    cells = [(7, 8, 9), (0, 0, 0)]
    log_skipped_mutation(
        step_index=20,
        suppressed_cells=cells,
        reason="tagging suppressed",
        output_folder=str(folder)
    )

    log_file = folder / "mutation_pathways_log.json"
    with open(log_file) as f:
        data = json.load(f)

    entry = data[-1]
    assert entry["step_index"] == 20
    assert entry["pressure_mutated"] is False
    assert entry["triggered_by"] == ["mutation suppressed"]
    assert "suppressed" in entry
    assert entry["suppressed"][0]["pressure_changed"] is True
    assert entry["suppressed"][0]["suppression_reason"] == "tagging suppressed"

def test_serialize_cell_with_object():
    cell = make_cell(9, 9, 9)
    cell.triggered_by = "ghost zone"
    cell.ghost_influence_attempted = True
    cell.ghost_influence_applied = False

    result = serialize_cell(cell)
    assert result["x"] == 9
    assert result["y"] == 9
    assert result["z"] == 9
    assert result["pressure_changed"] is False
    assert result["triggered_by"] == "ghost zone"
    assert result["influence_attempted"] is True
    assert result["influence_applied"] is False

def test_serialize_cell_with_tuple_and_reason():
    result = serialize_cell((5, 6, 7), reason="proximity cutoff")
    assert result["x"] == 5
    assert result["y"] == 6
    assert result["z"] == 7
    assert result["pressure_changed"] is True
    assert result["suppression_reason"] == "proximity cutoff"



