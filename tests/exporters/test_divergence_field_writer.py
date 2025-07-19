# tests/exporters/test_divergence_field_writer.py
# 🧪 Unit tests for src/exporters/divergence_field_writer.py

import os
import json
import tempfile
from src.exporters import divergence_field_writer

def test_export_creates_correct_file_structure():
    divergence_map = {
        (0.0, 0.0, 0.0): {"pre": 0.123456789, "post": 0.987654321},
        (1.5, 2.0, -0.5): {"pre": 0.1, "post": 0.2}
    }
    tmpdir = tempfile.mkdtemp()
    step = 3

    divergence_field_writer.export_divergence_map(divergence_map, step_index=step, output_dir=tmpdir)
    expected_file = os.path.join(tmpdir, "divergence_map_step_0003.json")
    assert os.path.isfile(expected_file)

    with open(expected_file) as f:
        data = json.load(f)
    
    assert "(0.00, 0.00, 0.00)" in data
    assert "(1.50, 2.00, -0.50)" in data
    assert data["(0.00, 0.00, 0.00)"]["divergence_before"] == round(0.123456789, 6)
    assert data["(0.00, 0.00, 0.00)"]["divergence_after"] == round(0.987654321, 6)

def test_export_creates_directory_if_missing():
    divergence_map = {(0.0, 0.0, 0.0): {"pre": 1.0, "post": 2.0}}
    tmpdir = tempfile.mkdtemp()
    nested = os.path.join(tmpdir, "nested", "snapshots")

    divergence_field_writer.export_divergence_map(divergence_map, step_index=0, output_dir=nested)
    expected_file = os.path.join(nested, "divergence_map_step_0000.json")
    assert os.path.isfile(expected_file)

def test_export_rounds_values_to_six_decimal_places():
    divergence_map = {
        (1.1, 2.2, 3.3): {"pre": 0.123456789, "post": 0.987654321}
    }
    tmpdir = tempfile.mkdtemp()

    divergence_field_writer.export_divergence_map(divergence_map, step_index=5, output_dir=tmpdir)
    expected_file = os.path.join(tmpdir, "divergence_map_step_0005.json")

    with open(expected_file) as f:
        data = json.load(f)

    entry = data["(1.10, 2.20, 3.30)"]
    assert entry["divergence_before"] == 0.123457
    assert entry["divergence_after"] == 0.987654

def test_export_handles_missing_keys_gracefully():
    divergence_map = {(0.0, 0.0, 0.0): {}}
    tmpdir = tempfile.mkdtemp()

    divergence_field_writer.export_divergence_map(divergence_map, step_index=2, output_dir=tmpdir)
    expected_file = os.path.join(tmpdir, "divergence_map_step_0002.json")

    with open(expected_file) as f:
        data = json.load(f)
    
    assert data["(0.00, 0.00, 0.00)"]["divergence_before"] == 0.0
    assert data["(0.00, 0.00, 0.00)"]["divergence_after"] == 0.0



