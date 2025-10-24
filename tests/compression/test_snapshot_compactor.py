# tests/compression/test_snapshot_compactor.py
# ✅ Validation suite for src/compression/snapshot_compactor.py

import os
import json
import pytest
from src.compression.snapshot_compactor import compact_pressure_delta_map

def test_compaction_retains_cells_above_threshold(tmp_path, capsys):
    input_path = tmp_path / "delta_map.json"
    output_path = tmp_path / "compacted.json"

    data = {
        "(0,0,0)": {"delta": 1e-5},
        "(0,0,1)": {"delta": 2e-8},
        "(0,0,2)": {"delta": 1e-9},  # below threshold
        "(0,0,3)": {"delta": 0.0},   # zero
    }
    with open(input_path, "w") as f:
        json.dump(data, f)

    retained_count = compact_pressure_delta_map(str(input_path), str(output_path), mutation_threshold=1e-8)
    assert retained_count == 2
    assert output_path.exists()

    with open(output_path) as f:
        compacted = json.load(f)
    assert "(0,0,0)" in compacted
    assert "(0,0,1)" in compacted
    assert "(0,0,2)" not in compacted
    assert "(0,0,3)" not in compacted

    captured = capsys.readouterr()
    assert "✅ Compacted snapshot saved" in captured.out
    assert "🧮 Cells retained: 2 of 4" in captured.out

def test_compaction_skips_write_if_nothing_retained(tmp_path, capsys):
    input_path = tmp_path / "delta_map.json"
    output_path = tmp_path / "compacted.json"

    data = {
        "(1,1,1)": {"delta": 1e-10},
        "(1,1,2)": {"delta": 0.0}
    }
    with open(input_path, "w") as f:
        json.dump(data, f)

    retained_count = compact_pressure_delta_map(str(input_path), str(output_path), mutation_threshold=1e-8)
    assert retained_count == 0
    assert not output_path.exists()

    captured = capsys.readouterr()
    assert "⚠️ No cells retained after compaction" in captured.out

def test_compaction_handles_missing_file_gracefully(tmp_path, capsys):
    input_path = tmp_path / "missing.json"
    output_path = tmp_path / "compacted.json"

    retained_count = compact_pressure_delta_map(str(input_path), str(output_path))
    assert retained_count == 0
    assert not output_path.exists()

    captured = capsys.readouterr()
    assert "❌ Failed to load" in captured.out

def test_compaction_handles_empty_file(tmp_path, capsys):
    input_path = tmp_path / "empty.json"
    output_path = tmp_path / "compacted.json"
    input_path.write_text("")

    retained_count = compact_pressure_delta_map(str(input_path), str(output_path))
    assert retained_count == 0
    assert not output_path.exists()

    captured = capsys.readouterr()
    assert "❌ Failed to load" in captured.out

def test_compaction_creates_output_directory(tmp_path):
    input_path = tmp_path / "delta_map.json"
    output_dir = tmp_path / "nested" / "output"
    output_path = output_dir / "compacted.json"

    data = {
        "(2,2,2)": {"delta": 1e-4}
    }
    with open(input_path, "w") as f:
        json.dump(data, f)

    retained_count = compact_pressure_delta_map(str(input_path), str(output_path))
    assert retained_count == 1
    assert output_path.exists()

    with open(output_path) as f:
        compacted = json.load(f)
    assert "(2,2,2)" in compacted



