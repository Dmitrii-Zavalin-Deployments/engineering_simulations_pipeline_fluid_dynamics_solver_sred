# tests/test_cell_builder_cli.py
# ✅ Tests for __main__ block of cell_builder.py

import json
import subprocess
import sys
import tempfile
import os
import pytest

SCRIPT = "src/step_1_solver_initialization/cell_builder.py"

def make_minimal_config():
    return {
        "domain_definition": {"nx": 1, "ny": 1, "nz": 1},
        "geometry_definition": {
            "geometry_mask_flat": [0],
            "mask_encoding": {"fluid": 0, "solid": 1, "boundary": 2},
        },
        "initial_conditions": {
            "initial_pressure": 100.0,
            "initial_velocity": (1.0, 2.0, 3.0),
        },
        "boundary_conditions": [],
    }

def test_cli_success(tmp_path):
    # Write input config
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    with open(input_file, "w") as f:
        json.dump(make_minimal_config(), f)

    # Run the script
    result = subprocess.run(
        [sys.executable, SCRIPT, "--input", str(input_file), "--output", str(output_file)],
        capture_output=True,
        text=True,
    )

    # Should succeed
    assert result.returncode == 0
    assert output_file.exists()

    # Validate output JSON
    with open(output_file) as f:
        data = json.load(f)
    assert 0 in data
    assert data[0]["cell_type"] == "fluid"

def test_cli_missing_input(tmp_path):
    output_file = tmp_path / "output.json"

    # Run with non-existent input
    result = subprocess.run(
        [sys.executable, SCRIPT, "--input", "no_such.json", "--output", str(output_file)],
        capture_output=True,
        text=True,
    )

    # Should fail
    assert result.returncode == 1
    assert "❌ Error running cell_builder" in result.stderr



