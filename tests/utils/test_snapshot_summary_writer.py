# ✅ Unit Test Suite — Snapshot Summary Writer
# 📄 Full Path: tests/utils/test_snapshot_summary_writer.py

import pytest
import os
import csv
from tempfile import TemporaryDirectory
from src.utils.snapshot_summary_writer import write_step_summary

def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def test_summary_file_created_and_header_written():
    with TemporaryDirectory() as tmp:
        meta = {
            "reflex_score": 0.75,
            "adaptive_timestep": 0.01,
            "mutation_density": 0.4,
            "pressure_mutated": True,
            "projection_passes": 2,
            "ghost_influence_count": 3,
            "post_projection_divergence": 0.02,
            "mean_divergence": 0.005,
            "fluid_cells_modified_by_ghost": 0
        }
        path = os.path.join(tmp, "snapshot_summary.csv")
        write_step_summary(1, meta, output_folder=tmp)
        assert os.path.exists(path)
        rows = read_csv(path)
        assert len(rows) == 1
        assert rows[0]["step_index"] == "1"
        assert rows[0]["ghost_adjacent_but_influence_suppressed"] == "True"

def test_summary_appends_multiple_steps():
    with TemporaryDirectory() as tmp:
        for i in range(1, 4):
            meta = {
                "reflex_score": 0.5 + 0.1 * i,
                "pressure_mutated": i % 2 == 0,
                "projection_passes": i,
                "ghost_influence_count": i,
                "fluid_cells_modified_by_ghost": i - 1,
                "post_projection_divergence": i * 0.01,
                "mean_divergence": i * 0.001
            }
            write_step_summary(i, meta, output_folder=tmp)
        rows = read_csv(os.path.join(tmp, "snapshot_summary.csv"))
        assert len(rows) == 3
        assert rows[0]["step_index"] == "1"
        assert rows[-1]["step_index"] == "3"

def test_score_casting_and_missing_fields_handled():
    with TemporaryDirectory() as tmp:
        meta = {
            "reflex_score": "bad_value",  # should fallback to 0.0
            "ghost_influence_count": 1,
            "fluid_cells_modified_by_ghost": 0
        }
        write_step_summary(5, meta, output_folder=tmp)
        rows = read_csv(os.path.join(tmp, "snapshot_summary.csv"))
        assert rows[0]["reflex_score"] == "0.0"
        assert rows[0]["ghost_adjacent_but_influence_suppressed"] == "True"
        assert rows[0]["adaptive_timestep"] == ""

def test_suppression_flag_false_when_influence_exists():
    with TemporaryDirectory() as tmp:
        meta = {
            "ghost_influence_count": 2,
            "fluid_cells_modified_by_ghost": 1
        }
        write_step_summary(10, meta, output_folder=tmp)
        rows = read_csv(os.path.join(tmp, "snapshot_summary.csv"))
        assert rows[0]["ghost_adjacent_but_influence_suppressed"] == "False"



