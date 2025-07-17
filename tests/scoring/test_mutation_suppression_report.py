# Filename: tests/scoring/test_mutation_suppression_report.py

import json
import pytest
import os
from scoring.mutation_suppression_report import export_suppression_report

@pytest.fixture
def temp_output_dir(tmp_path):
    return str(tmp_path)

def test_export_report_with_suppressed_data(temp_output_dir):
    step_index = 10
    suppressed_cells = [
        {"cell": [0.5, 0.75, 0.5], "reason": "ghost matched fields"},
        {"cell": [1.0, 0.25, 0.5], "reason": "velocity identical"}
    ]

    export_suppression_report(step_index, suppressed_cells, temp_output_dir)

    expected_file = os.path.join(temp_output_dir, "mutation_suppression_step_0010.json")
    assert os.path.isfile(expected_file)

    with open(expected_file) as f:
        data = json.load(f)

    assert data["step"] == 10
    assert len(data["suppressed"]) == 2
    assert data["suppressed"][0]["reason"] == "ghost matched fields"

def test_export_skips_empty_suppressed_list(temp_output_dir, caplog):
    step_index = 5
    suppressed_cells = []

    export_suppression_report(step_index, suppressed_cells, temp_output_dir)

    expected_file = os.path.join(temp_output_dir, "mutation_suppression_step_0005.json")
    assert not os.path.exists(expected_file)
    assert f"No suppressed mutations for step {step_index}" in caplog.text



