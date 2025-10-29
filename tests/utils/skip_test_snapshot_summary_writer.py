import csv
import os
import pytest
from src.utils.snapshot_summary_writer import write_step_summary

@pytest.fixture
def reflex_metadata_full():
    return {
        "reflex_score": 0.92,
        "adaptive_timestep": 0.004,
        "mutation_density": 0.15,
        "pressure_mutated": True,
        "projection_passes": 3,
        "ghost_influence_count": 2,
        "fluid_cells_modified_by_ghost": 0,
        "adjacency_zones": [(0, 0, 0), (1, 1, 1)],
        "boundary_cell_count": 12,
        "boundary_mutation_ratio": 0.25,
        "post_projection_divergence": 0.003,
        "mean_divergence": 0.001,
        "mutated_cells": [{"x": 0, "y": 0, "z": 0}],
        "damping_triggered_count": 1,
        "overflow_triggered_count": 0,
        "cfl_exceeded_count": 2
    }

@pytest.fixture
def reflex_metadata_minimal():
    return {
        "reflex_score": "invalid",
        "mutated_cells": [],
        "ghost_influence_count": 0,
        "fluid_cells_modified_by_ghost": 0
    }

@pytest.fixture
def summary_path(tmp_path):
    return tmp_path / "snapshot_summary.csv"

def test_write_step_summary_full(tmp_path, reflex_metadata_full):
    write_step_summary(step_index=1, reflex_metadata=reflex_metadata_full, output_folder=str(tmp_path))
    summary_file = tmp_path / "snapshot_summary.csv"
    assert summary_file.exists()

    with open(summary_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["step_index"] == "1"
        assert row["reflex_score"] == "0.92"
        assert row["ghost_adjacent_but_influence_suppressed"] == "True"
        assert row["mutation_count"] == "1"
        assert row["boundary_mutation_ratio"] == "0.25"

def test_write_step_summary_minimal(tmp_path, reflex_metadata_minimal):
    write_step_summary(step_index=2, reflex_metadata=reflex_metadata_minimal, output_folder=str(tmp_path))
    summary_file = tmp_path / "snapshot_summary.csv"
    assert summary_file.exists()

    with open(summary_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["step_index"] == "2"
        assert row["reflex_score"] == "0.0"
        assert row["ghost_adjacent_but_influence_suppressed"] == "False"
        assert row["mutation_count"] == "0"

def test_write_step_summary_appends(tmp_path, reflex_metadata_full, reflex_metadata_minimal):
    write_step_summary(step_index=3, reflex_metadata=reflex_metadata_full, output_folder=str(tmp_path))
    write_step_summary(step_index=4, reflex_metadata=reflex_metadata_minimal, output_folder=str(tmp_path))

    summary_file = tmp_path / "snapshot_summary.csv"
    with open(summary_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["step_index"] == "3"
        assert rows[1]["step_index"] == "4"

def test_write_step_summary_creates_directory(tmp_path, reflex_metadata_full):
    custom_folder = tmp_path / "nested" / "summaries"
    write_step_summary(step_index=5, reflex_metadata=reflex_metadata_full, output_folder=str(custom_folder))
    summary_file = custom_folder / "snapshot_summary.csv"
    assert summary_file.exists()



