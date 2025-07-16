# tests/exporters/test_pressure_delta_map_writer.py
# ✅ Unit tests for pressure_delta_map_writer

import os
import json
import shutil
import pytest
from src.exporters.pressure_delta_map_writer import export_pressure_delta_map

@pytest.fixture
def mock_output_dir(tmp_path):
    return tmp_path / "snapshots"

def test_export_creates_expected_file(mock_output_dir):
    pressure_delta_map = {
        (0.5, 0.25, 0.75): {"before": 18.0, "after": 20.0, "delta": 2.0},
        (1.0, 0.5, 1.0): {"before": 19.5, "after": 19.5, "delta": 0.0}
    }
    step_index = 4

    export_pressure_delta_map(pressure_delta_map, step_index, str(mock_output_dir))

    expected_file = mock_output_dir / "pressure_delta_map_step_0004.json"
    assert expected_file.exists(), "Output file was not created"

    with open(expected_file, "r") as f:
        data = json.load(f)

    assert "(0.50, 0.25, 0.75)" in data
    assert "(1.00, 0.50, 1.00)" in data
    assert data["(0.50, 0.25, 0.75)"]["pressure_before"] == 18.0
    assert data["(0.50, 0.25, 0.75)"]["pressure_after"] == 20.0
    assert data["(0.50, 0.25, 0.75)"]["delta"] == 2.0

def test_export_with_empty_data(mock_output_dir):
    pressure_delta_map = {}
    step_index = 7

    export_pressure_delta_map(pressure_delta_map, step_index, str(mock_output_dir))

    expected_file = mock_output_dir / "pressure_delta_map_step_0007.json"
    assert expected_file.exists()

    with open(expected_file, "r") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert len(data) == 0

def test_export_with_partial_fields(mock_output_dir):
    pressure_delta_map = {
        (2.5, 0.25, 0.75): {"before": 20.0},
        (3.0, 0.5, 1.0): {"after": 21.0},
        (3.5, 0.5, 1.5): {}  # all missing
    }
    step_index = 12

    export_pressure_delta_map(pressure_delta_map, step_index, str(mock_output_dir))

    expected_file = mock_output_dir / "pressure_delta_map_step_0012.json"
    assert expected_file.exists()

    with open(expected_file, "r") as f:
        data = json.load(f)

    assert data["(2.50, 0.25, 0.75)"]["pressure_before"] == 20.0
    assert data["(2.50, 0.25, 0.75)"]["pressure_after"] == 0.0
    assert data["(3.50, 0.50, 1.50)"]["delta"] == 0.0

def test_export_to_nonexistent_directory(tmp_path):
    target_dir = tmp_path / "new_outputs" / "nested" / "snapshots"
    pressure_delta_map = {
        (0.0, 0.0, 0.0): {"before": 10.0, "after": 12.0, "delta": 2.0}
    }
    step_index = 1

    export_pressure_delta_map(pressure_delta_map, step_index, str(target_dir))

    output_file = target_dir / "pressure_delta_map_step_0001.json"
    assert output_file.exists()

    with open(output_file, "r") as f:
        content = json.load(f)
        assert "(0.00, 0.00, 0.00)" in content



