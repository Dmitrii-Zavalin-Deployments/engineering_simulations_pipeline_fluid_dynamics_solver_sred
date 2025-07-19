# tests/scoring/test_mutation_suppression_report.py
# 🧪 Unit tests for src/scoring/mutation_suppression_report.py

import os
import json
import tempfile
import logging
import pytest
from src.scoring.mutation_suppression_report import export_suppression_report

def test_report_skipped_when_suppressed_cells_empty(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        with caplog.at_level(logging.DEBUG):
            export_suppression_report(step_index=3, suppressed_cells=[], output_dir=tmp)
        assert "[report] No suppressed mutations for step 3" in caplog.text
        files = os.listdir(tmp)
        assert len(files) == 0

def test_report_file_created_with_expected_content():
    with tempfile.TemporaryDirectory() as tmp:
        suppressed = [
            {"cell": [1.0, 2.0, 3.0], "reason": "velocity match"},
            {"cell": [4.0, 5.0, 6.0], "reason": "pressure match"}
        ]
        export_suppression_report(step_index=12, suppressed_cells=suppressed, output_dir=tmp)
        expected_path = os.path.join(tmp, "mutation_suppression_step_0012.json")
        assert os.path.exists(expected_path)
        with open(expected_path) as f:
            report = json.load(f)
        assert report["step"] == 12
        assert len(report["suppressed"]) == 2
        assert report["suppressed"][0]["cell"] == [1.0, 2.0, 3.0]
        assert report["suppressed"][1]["reason"] == "pressure match"

def test_directory_created_if_missing():
    with tempfile.TemporaryDirectory() as base:
        subdir = os.path.join(base, "output/subdir")
        suppressed = [{"cell": [0.0, 0.0, 0.0], "reason": "adjacency flag"}]
        export_suppression_report(step_index=7, suppressed_cells=suppressed, output_dir=subdir)
        file_path = os.path.join(subdir, "mutation_suppression_step_0007.json")
        assert os.path.exists(file_path)



