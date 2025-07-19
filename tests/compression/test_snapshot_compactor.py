# tests/compression/test_snapshot_compactor.py
# 🧪 Unit tests for src/compression/snapshot_compactor.py

import os
import json
import tempfile
from src.compression import snapshot_compactor

def make_snapshot_file(data: dict) -> str:
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        json.dump(data, f)
        return f.name

def make_output_path() -> str:
    tmpdir = tempfile.mkdtemp()
    return os.path.join(tmpdir, "compacted_snapshot.json")

def test_compactor_retains_high_delta_cells():
    data = {
        "(1.0,1.0,1.0)": {"delta": 0.5},
        "(2.0,2.0,2.0)": {"delta": 0.0},
        "(3.0,3.0,3.0)": {"delta": 1e-6},
    }
    input_path = make_snapshot_file(data)
    output_path = make_output_path()

    count = snapshot_compactor.compact_pressure_delta_map(
        input_path, output_path, mutation_threshold=1e-8
    )
    assert count == 2

    with open(output_path) as f:
        compacted = json.load(f)
    assert "(1.0,1.0,1.0)" in compacted
    assert "(3.0,3.0,3.0)" in compacted
    assert "(2.0,2.0,2.0)" not in compacted

def test_compactor_removes_all_if_below_threshold():
    data = {
        "(0.0,0.0,0.0)": {"delta": 1e-10},
        "(1.0,1.0,1.0)": {"delta": -1e-9},
    }
    input_path = make_snapshot_file(data)
    output_path = make_output_path()

    count = snapshot_compactor.compact_pressure_delta_map(
        input_path, output_path, mutation_threshold=1e-8
    )
    assert count == 0
    assert not os.path.isfile(output_path)

def test_compactor_handles_missing_file_gracefully():
    bad_path = "/tmp/nonexistent_file.json"
    output_path = make_output_path()

    count = snapshot_compactor.compact_pressure_delta_map(
        bad_path, output_path
    )
    assert count == 0
    assert not os.path.isfile(output_path)

def test_compactor_creates_directory_if_missing():
    data = {
        "(0.0,0.0,0.0)": {"delta": 1.0},
    }
    input_path = make_snapshot_file(data)
    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "nested", "compacted.json")

    count = snapshot_compactor.compact_pressure_delta_map(input_path, output_path)
    assert count == 1
    assert os.path.isfile(output_path)

def test_compactor_retains_exact_threshold_values():
    data = {
        "(0.5,0.5,0.5)": {"delta": 1e-8},
        "(1.0,1.0,1.0)": {"delta": 2e-8},
    }
    input_path = make_snapshot_file(data)
    output_path = make_output_path()

    count = snapshot_compactor.compact_pressure_delta_map(
        input_path, output_path, mutation_threshold=1e-8
    )
    assert count == 1
    with open(output_path) as f:
        compacted = json.load(f)
    assert "(1.0,1.0,1.0)" in compacted
    assert "(0.5,0.5,0.5)" not in compacted



