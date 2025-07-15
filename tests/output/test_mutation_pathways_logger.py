# tests/test_mutation_pathways_logger.py
# 🧪 Unit tests for mutation_pathways_logger.py — verifies serialization, log structure, and edge handling

import os
import json
import shutil
import pytest
from src.output.mutation_pathways_logger import serialize_cell, log_mutation_pathway
from src.grid_modules.cell import Cell

OUTPUT_DIR = "data/testing-output/mutation_pathway_tests"

@pytest.fixture
def clean_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    yield OUTPUT_DIR
    shutil.rmtree(OUTPUT_DIR)

def test_serialize_cell_from_dataclass():
    cell = Cell(x=1.0, y=2.0, z=3.0, velocity=[1.0, 0.0, 0.0], pressure=50.0, fluid_mask=True)
    result = serialize_cell(cell)
    assert result["x"] == 1.0
    assert result["velocity"] == [1.0, 0.0, 0.0]
    assert result["fluid_mask"] is True

def test_serialize_cell_from_tuple():
    result = serialize_cell((4.0, 5.0, 6.0))
    assert result == {"x": 4.0, "y": 5.0, "z": 6.0}

def test_log_mutation_pathway_basic(clean_output_folder):
    log_mutation_pathway(
        step_index=0,
        pressure_mutated=True,
        triggered_by=["advection", "viscosity"],
        output_folder=clean_output_folder
    )
    path = os.path.join(clean_output_folder, "mutation_pathways_log.json")
    assert os.path.exists(path)
    with open(path) as f:
        entries = json.load(f)
    assert isinstance(entries, list)
    assert entries[0]["step_index"] == 0
    assert "triggered_by" in entries[0]

def test_log_mutation_with_triggered_cells(clean_output_folder):
    cells = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0, 0, 0], pressure=0.0, fluid_mask=True),
        (1.0, 1.0, 1.0)
    ]
    log_mutation_pathway(
        step_index=1,
        pressure_mutated=False,
        triggered_by=["reflex_skip"],
        triggered_cells=cells,
        output_folder=clean_output_folder
    )
    path = os.path.join(clean_output_folder, "mutation_pathways_log.json")
    with open(path) as f:
        entries = json.load(f)
    entry = next(e for e in entries if e["step_index"] == 1)
    assert len(entry["triggered_cells"]) == 2
    assert entry["triggered_cells"][1]["x"] == 1.0

def test_log_mutation_pathway_appends_multiple(clean_output_folder):
    for i in range(3):
        log_mutation_pathway(
            step_index=i,
            pressure_mutated=bool(i % 2),
            triggered_by=["stage" + str(i)],
            output_folder=clean_output_folder
        )
    path = os.path.join(clean_output_folder, "mutation_pathways_log.json")
    with open(path) as f:
        entries = json.load(f)
    assert len(entries) == 3
    assert entries[2]["step_index"] == 2

def test_log_handles_corrupt_json_file(clean_output_folder):
    # Write junk to file
    path = os.path.join(clean_output_folder, "mutation_pathways_log.json")
    with open(path, "w") as f:
        f.write("💥 NOT JSON 💥")
    # Should recover and overwrite
    log_mutation_pathway(
        step_index=99,
        pressure_mutated=False,
        triggered_by=["corruption-recovery"],
        output_folder=clean_output_folder
    )
    with open(path) as f:
        entries = json.load(f)
    assert isinstance(entries, list)
    assert entries[-1]["step_index"] == 99

def test_serialization_failure_raises_typeerror(clean_output_folder):
    class BadCell:
        def __init__(self):
            self.x = 1.0
            self.y = 2.0
            self.z = 3.0
            self.velocity = set([1.0])  # sets can't be serialized

    with pytest.raises(TypeError):
        log_mutation_pathway(
            step_index=2,
            pressure_mutated=True,
            triggered_by=["bad_velocity"],
            triggered_cells=[BadCell()],
            output_folder=clean_output_folder
        )