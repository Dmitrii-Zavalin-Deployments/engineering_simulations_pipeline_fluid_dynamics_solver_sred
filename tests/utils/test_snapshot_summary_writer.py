# tests/utils/test_snapshot_summary_writer.py
# ✅ Unit tests for snapshot_summary_writer.py

import csv
import os
import pytest
from src.utils.snapshot_summary_writer import write_step_summary

@pytest.fixture
def summary_path(tmp_path):
    return tmp_path / "snapshot_summary.csv"

def test_write_single_summary_row(summary_path):
    reflex_metadata = {
        "reflex_score": 5,
        "adaptive_timestep": 0.008,
        "mutation_density": 0.25,
        "pressure_mutated": True,
        "projection_passes": 2,
        "ghost_influence_count": 7,
        "post_projection_divergence": 0.004,
        "mean_divergence": 0.0015
    }
    write_step_summary(3, reflex_metadata, output_folder=summary_path.parent)
    assert summary_path.exists()

    with open(summary_path, "r") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    row = reader[0]
    assert row["step_index"] == "3"
    assert row["reflex_score"] == "5"
    assert row["adaptive_timestep"] == "0.008"
    assert row["mutation_density"] == "0.25"

def test_write_multiple_rows(summary_path):
    for i in range(5):
        write_step_summary(i, {"reflex_score": i}, output_folder=summary_path.parent)

    with open(summary_path, "r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert rows[4]["step_index"] == "4"
    assert rows[4]["reflex_score"] == "4"

def test_missing_fields_handled_gracefully(summary_path):
    write_step_summary(1, {}, output_folder=summary_path.parent)
    with open(summary_path, "r") as f:
        row = next(csv.DictReader(f))
    assert row["step_index"] == "1"
    assert row["reflex_score"] == ""

def test_output_folder_creation(tmp_path):
    target_dir = tmp_path / "nested" / "summary"
    write_step_summary(2, {"reflex_score": 4}, output_folder=str(target_dir))
    path = target_dir / "snapshot_summary.csv"
    assert path.exists()