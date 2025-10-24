# tests/exporters/test_pressure_delta_map_writer.py
# ✅ Validation suite for src/exporters/pressure_delta_map_writer.py

import os
import json
import tempfile
from src.exporters.pressure_delta_map_writer import export_pressure_delta_map

def test_export_creates_expected_file_and_directory():
    pressure_delta_map = [
        {"x": 0, "y": 0, "z": 0, "before": 101325.123456, "after": 101325.654321, "delta": 0.530865},
        {"x": 1, "y": 2, "z": 3, "before": 100000.987654, "after": 100001.111111, "delta": 0.123457}
    ]

    with tempfile.TemporaryDirectory() as tmp:
        export_pressure_delta_map(pressure_delta_map, step_index=42, output_dir=tmp)
        expected_file = os.path.join(tmp, "pressure_delta_map_step_0042.json")
        assert os.path.exists(expected_file)

        with open(expected_file) as f:
            data = json.load(f)

        assert "(0.00, 0.00, 0.00)" in data
        assert "(1.00, 2.00, 3.00)" in data
        assert data["(0.00, 0.00, 0.00)"]["pressure_before"] == round(101325.123456, 6)
        assert data["(0.00, 0.00, 0.00)"]["pressure_after"] == round(101325.654321, 6)
        assert data["(0.00, 0.00, 0.00)"]["delta"] == round(0.530865, 6)

def test_export_handles_missing_keys_gracefully():
    pressure_delta_map = [
        {"x": 5, "y": 5, "z": 5},  # no 'before', 'after', 'delta'
        {"x": 1, "y": 1, "z": 1, "before": 101000.0}  # missing 'after' and 'delta'
    ]

    with tempfile.TemporaryDirectory() as tmp:
        export_pressure_delta_map(pressure_delta_map, step_index=7, output_dir=tmp)
        expected_file = os.path.join(tmp, "pressure_delta_map_step_0007.json")
        with open(expected_file) as f:
            data = json.load(f)

        assert data["(5.00, 5.00, 5.00)"]["pressure_before"] == 0.0
        assert data["(5.00, 5.00, 5.00)"]["pressure_after"] == 0.0
        assert data["(5.00, 5.00, 5.00)"]["delta"] == 0.0

        assert data["(1.00, 1.00, 1.00)"]["pressure_before"] == 101000.0
        assert data["(1.00, 1.00, 1.00)"]["pressure_after"] == 0.0
        assert data["(1.00, 1.00, 1.00)"]["delta"] == 0.0

def test_export_rounding_precision():
    pressure_delta_map = [
        {"x": 2, "y": 2, "z": 2, "before": 123456.7890123, "after": 123456.7899876, "delta": 0.0009753}
    ]

    with tempfile.TemporaryDirectory() as tmp:
        export_pressure_delta_map(pressure_delta_map, step_index=99, output_dir=tmp)
        expected_file = os.path.join(tmp, "pressure_delta_map_step_0099.json")
        with open(expected_file) as f:
            data = json.load(f)

        assert data["(2.00, 2.00, 2.00)"]["pressure_before"] == round(123456.7890123, 6)
        assert data["(2.00, 2.00, 2.00)"]["pressure_after"] == round(123456.7899876, 6)
        assert data["(2.00, 2.00, 2.00)"]["delta"] == round(0.0009753, 6)

def test_export_default_output_dir(tmp_path):
    pressure_delta_map = [
        {"x": 0, "y": 0, "z": 0, "before": 0.0, "after": 0.0, "delta": 0.0}
    ]

    # Use default output_dir = "data/snapshots"
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # simulate working directory
        export_pressure_delta_map(pressure_delta_map, step_index=1)
        expected_file = os.path.join("data", "snapshots", "pressure_delta_map_step_0001.json")
        assert os.path.exists(expected_file)

def test_export_empty_map_creates_valid_file():
    pressure_delta_map = []

    with tempfile.TemporaryDirectory() as tmp:
        export_pressure_delta_map(pressure_delta_map, step_index=0, output_dir=tmp)
        expected_file = os.path.join(tmp, "pressure_delta_map_step_0000.json")
        assert os.path.exists(expected_file)

        with open(expected_file) as f:
            data = json.load(f)
        assert data == {}



