import os
import json
import shutil
import pytest
from src.reflex.reflex_pathway_logger import log_reflex_pathway, serialize_mutation_cell
from src.grid_modules.cell import Cell

@pytest.fixture
def temp_output_folder(tmp_path):
    return tmp_path / "navier_stokes_output"

def make_cell(x, y, z, fluid_mask=True, pressure_mutated=False, mutation_triggered_by=None, mutation_source=None, pressure_delta=None):
    cell = Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=fluid_mask)
    cell.pressure_mutated = pressure_mutated
    cell.mutation_triggered_by = mutation_triggered_by
    cell.mutation_source = mutation_source
    cell.pressure_delta = pressure_delta
    return cell

def test_serialize_mutation_cell_with_cell_object():
    cell = make_cell(1.0, 2.0, 3.0, pressure_mutated=True, mutation_triggered_by="ghost", mutation_source="step_0", pressure_delta=0.05)
    result = serialize_mutation_cell(cell, step_linked_from=0)
    assert result["x"] == 1.0
    assert result["y"] == 2.0
    assert result["z"] == 3.0
    assert result["step_linked_from"] == 0
    assert result["pressure_mutated"] is True
    assert result["mutation_triggered_by"] == "ghost"
    assert result["mutation_source"] == "step_0"
    assert result["pressure_delta"] == 0.05

def test_serialize_mutation_cell_with_tuple():
    result = serialize_mutation_cell((4.0, 5.0, 6.0), step_linked_from=1)
    assert result["x"] == 4.0
    assert result["y"] == 5.0
    assert result["z"] == 6.0
    assert result["step_linked_from"] == 1
    assert result["pressure_mutated"] is False

def test_log_reflex_pathway_creates_file(temp_output_folder):
    cell = make_cell(0.0, 0.0, 0.0, pressure_mutated=True)
    log_reflex_pathway(step_index=1, mutated_cells=[cell], ghost_trigger_chain=[0], output_folder=str(temp_output_folder))
    log_path = temp_output_folder / "reflex_pathway_log.json"
    assert log_path.exists()
    with open(log_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert data[-1]["step_index"] == 1
    assert data[-1]["ghost_trigger_chain"] == [0]
    assert len(data[-1]["mutated_cells"]) == 1
    assert data[-1]["mutated_cells"][0]["x"] == 0.0

def test_log_reflex_pathway_appends_to_existing_log(temp_output_folder):
    cell1 = make_cell(0.0, 0.0, 0.0)
    cell2 = make_cell(1.0, 1.0, 1.0)
    log_reflex_pathway(step_index=1, mutated_cells=[cell1], output_folder=str(temp_output_folder))
    log_reflex_pathway(step_index=2, mutated_cells=[cell2], output_folder=str(temp_output_folder))
    log_path = temp_output_folder / "reflex_pathway_log.json"
    with open(log_path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["step_index"] == 1
    assert data[1]["step_index"] == 2

def test_log_reflex_pathway_handles_invalid_json(temp_output_folder):
    log_path = temp_output_folder / "reflex_pathway_log.json"
    os.makedirs(temp_output_folder, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("corrupted")
    cell = make_cell(0.0, 0.0, 0.0)
    log_reflex_pathway(step_index=3, mutated_cells=[cell], output_folder=str(temp_output_folder))
    with open(log_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert data[-1]["step_index"] == 3

def test_log_reflex_pathway_verbose_output(temp_output_folder, capsys):
    cell = make_cell(0.0, 0.0, 0.0)
    log_reflex_pathway(step_index=4, mutated_cells=[cell], ghost_trigger_chain=[2], output_folder=str(temp_output_folder), verbose=True)
    captured = capsys.readouterr()
    assert "Reflex pathway recorded for step 4" in captured.out
    assert "Mutated cells: 1" in captured.out
    assert "Ghost trigger chain: [2]" in captured.out



