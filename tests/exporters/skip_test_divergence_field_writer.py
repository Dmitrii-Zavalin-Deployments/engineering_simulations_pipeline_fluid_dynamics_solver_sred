# tests/exporters/test_divergence_field_writer.py
# ✅ Validation suite for src/exporters/divergence_field_writer.py

import os
import json
import tempfile
from src.exporters.divergence_field_writer import export_divergence_map

def test_export_creates_expected_file_and_directory():
    divergence_map = {
        (0, 0, 0): {"pre": 0.000123456, "post": 0.000654321},
        (1, 2, 3): {"pre": -0.000987654, "post": 0.000111111}
    }

    with tempfile.TemporaryDirectory() as tmp:
        export_divergence_map(divergence_map, step_index=42, output_dir=tmp)
        expected_file = os.path.join(tmp, "divergence_map_step_0042.json")
        assert os.path.exists(expected_file)

        with open(expected_file) as f:
            data = json.load(f)

        assert "(0.00, 0.00, 0.00)" in data
        assert "(1.00, 2.00, 3.00)" in data
        assert data["(0.00, 0.00, 0.00)"]["divergence_before"] == round(0.000123456, 6)
        assert data["(0.00, 0.00, 0.00)"]["divergence_after"] == round(0.000654321, 6)

def test_export_handles_missing_keys_gracefully():
    divergence_map = {
        (5, 5, 5): {},  # no 'pre' or 'post'
        (1, 1, 1): {"pre": 0.1}  # missing 'post'
    }

    with tempfile.TemporaryDirectory() as tmp:
        export_divergence_map(divergence_map, step_index=7, output_dir=tmp)
        expected_file = os.path.join(tmp, "divergence_map_step_0007.json")
        with open(expected_file) as f:
            data = json.load(f)

        assert data["(5.00, 5.00, 5.00)"]["divergence_before"] == 0.0
        assert data["(5.00, 5.00, 5.00)"]["divergence_after"] == 0.0
        assert data["(1.00, 1.00, 1.00)"]["divergence_before"] == 0.1
        assert data["(1.00, 1.00, 1.00)"]["divergence_after"] == 0.0

def test_export_rounding_precision():
    divergence_map = {
        (2, 2, 2): {"pre": 0.123456789, "post": -0.987654321}
    }

    with tempfile.TemporaryDirectory() as tmp:
        export_divergence_map(divergence_map, step_index=99, output_dir=tmp)
        expected_file = os.path.join(tmp, "divergence_map_step_0099.json")
        with open(expected_file) as f:
            data = json.load(f)

        assert data["(2.00, 2.00, 2.00)"]["divergence_before"] == round(0.123456789, 6)
        assert data["(2.00, 2.00, 2.00)"]["divergence_after"] == round(-0.987654321, 6)

def test_export_default_output_dir(tmp_path):
    divergence_map = {
        (0, 0, 0): {"pre": 0.0, "post": 0.0}
    }

    # Use default output_dir = "data/snapshots"
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # simulate working directory
        export_divergence_map(divergence_map, step_index=1)
        expected_file = os.path.join("data", "snapshots", "divergence_map_step_0001.json")
        assert os.path.exists(expected_file)

def test_export_empty_map_creates_valid_file():
    divergence_map = {}

    with tempfile.TemporaryDirectory() as tmp:
        export_divergence_map(divergence_map, step_index=0, output_dir=tmp)
        expected_file = os.path.join(tmp, "divergence_map_step_0000.json")
        assert os.path.exists(expected_file)

        with open(expected_file) as f:
            data = json.load(f)
        assert data == {}



