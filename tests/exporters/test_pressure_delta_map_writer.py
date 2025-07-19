# tests/exporters/test_pressure_delta_map_writer.py
# 🧪 Unit tests for src/exporters/pressure_delta_map_writer.py

import os
import json
import tempfile
from src.exporters import pressure_delta_map_writer

def test_export_creates_snapshot_file_with_correct_data():
    delta_map = {
        (1.0, 2.0, 3.0): {"before": 10.123456789, "after": 20.987654321, "delta": 10.864197532},
        (0.0, 0.0, 0.0): {"before": 0.0, "after": 0.0, "delta": 0.0}
    }
    tmpdir = tempfile.mkdtemp()
    step_index = 5

    pressure_delta_map_writer.export_pressure_delta_map(delta_map, step_index, output_dir=tmpdir)
    expected_file = os.path.join(tmpdir, "pressure_delta_map_step_0005.json")
    assert os.path.isfile(expected_file)

    with open(expected_file) as f:
        data = json.load(f)
    
    assert data["(1.00, 2.00, 3.00)"]["pressure_before"] == round(10.123456789, 6)
    assert data["(1.00, 2.00, 3.00)"]["pressure_after"] == round(20.987654321, 6)
    assert data["(1.00, 2.00, 3.00)"]["delta"] == round(10.864197532, 6)

def test_export_rounds_values_to_six_decimals():
    delta_map = {
        (0.1, 0.1, 0.1): {"before": 1.000000123, "after": 2.000000789, "delta": 1.000000666}
    }
    tmpdir = tempfile.mkdtemp()

    pressure_delta_map_writer.export_pressure_delta_map(delta_map, step_index=1, output_dir=tmpdir)
    expected_file = os.path.join(tmpdir, "pressure_delta_map_step_0001.json")

    with open(expected_file) as f:
        data = json.load(f)
    
    values = data["(0.10, 0.10, 0.10)"]
    assert values["pressure_before"] == 1.000000
    assert values["pressure_after"] == 2.000001
    assert values["delta"] == 1.000001

def test_export_fallbacks_to_zero_when_keys_missing():
    delta_map = {
        (0.0, 0.0, 0.0): {}  # missing keys
    }
    tmpdir = tempfile.mkdtemp()

    pressure_delta_map_writer.export_pressure_delta_map(delta_map, step_index=2, output_dir=tmpdir)
    expected_file = os.path.join(tmpdir, "pressure_delta_map_step_0002.json")

    with open(expected_file) as f:
        data = json.load(f)
    
    assert data["(0.00, 0.00, 0.00)"]["pressure_before"] == 0.0
    assert data["(0.00, 0.00, 0.00)"]["pressure_after"] == 0.0
    assert data["(0.00, 0.00, 0.00)"]["delta"] == 0.0

def test_export_creates_output_directory_if_missing():
    delta_map = {
        (1.1, 2.2, 3.3): {"before": 1.0, "after": 2.0, "delta": 1.0}
    }
    tmpdir = tempfile.mkdtemp()
    target_dir = os.path.join(tmpdir, "nested", "snapshots")

    pressure_delta_map_writer.export_pressure_delta_map(delta_map, step_index=0, output_dir=target_dir)
    expected_file = os.path.join(target_dir, "pressure_delta_map_step_0000.json")
    assert os.path.isfile(expected_file)



