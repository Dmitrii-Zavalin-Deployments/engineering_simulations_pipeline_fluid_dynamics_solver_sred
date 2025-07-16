# tests/exporters/test_divergence_field_writer.py
# ✅ Unit tests for divergence_field_writer.py

import os
import json
import pytest
from src.exporters.divergence_field_writer import export_divergence_map

@pytest.fixture
def mock_output_dir(tmp_path):
    return tmp_path / "snapshots"

def test_export_divergence_map_creates_file(mock_output_dir):
    divergence_map = {
        (1.0, 0.5, 2.0): {"pre": 0.183083, "post": 0.162338},
        (0.5, 0.25, 1.5): {"pre": 0.200000, "post": 0.150000}
    }
    step_index = 9

    export_divergence_map(divergence_map, step_index, str(mock_output_dir))

    expected_file = mock_output_dir / "divergence_map_step_0009.json"
    assert expected_file.exists()

    with open(expected_file, "r") as f:
        data = json.load(f)

    assert "(1.00, 0.50, 2.00)" in data
    assert data["(1.00, 0.50, 2.00)"]["divergence_before"] == 0.183083
    assert data["(1.00, 0.50, 2.00)"]["divergence_after"] == 0.162338

def test_export_handles_empty_divergence_map(mock_output_dir):
    divergence_map = {}
    step_index = 10

    export_divergence_map(divergence_map, step_index, str(mock_output_dir))

    expected_file = mock_output_dir / "divergence_map_step_0010.json"
    assert expected_file.exists()

    with open(expected_file, "r") as f:
        content = json.load(f)
        assert isinstance(content, dict)
        assert len(content) == 0

def test_export_handles_missing_fields(mock_output_dir):
    divergence_map = {
        (2.0, 0.5, 1.0): {"pre": 0.100001},
        (3.0, 0.5, 1.5): {"post": 0.099999},
        (4.0, 0.75, 1.25): {}  # No values
    }
    step_index = 11

    export_divergence_map(divergence_map, step_index, str(mock_output_dir))

    expected_file = mock_output_dir / "divergence_map_step_0011.json"
    assert expected_file.exists()

    with open(expected_file, "r") as f:
        data = json.load(f)

    assert data["(2.00, 0.50, 1.00)"]["divergence_after"] == 0.0
    assert data["(3.00, 0.50, 1.50)"]["divergence_before"] == 0.0
    assert data["(4.00, 0.75, 1.25)"]["divergence_before"] == 0.0
    assert data["(4.00, 0.75, 1.25)"]["divergence_after"] == 0.0

def test_export_to_nested_directory(tmp_path):
    nested_dir = tmp_path / "metrics" / "steps"
    divergence_map = {
        (0.0, 0.0, 0.0): {"pre": 0.25, "post": 0.0}
    }
    step_index = 7

    export_divergence_map(divergence_map, step_index, str(nested_dir))

    expected_file = nested_dir / "divergence_map_step_0007.json"
    assert expected_file.exists()

    with open(expected_file, "r") as f:
        result = json.load(f)
        assert "(0.00, 0.00, 0.00)" in result
        assert result["(0.00, 0.00, 0.00)"]["divergence_after"] == 0.0



