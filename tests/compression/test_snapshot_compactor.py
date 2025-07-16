# tests/compression/test_snapshot_compactor.py
# ✅ Unit tests for snapshot_compactor.py

import json
import pytest
from src.compression.snapshot_compactor import compact_pressure_delta_map

@pytest.fixture
def delta_map_path(tmp_path):
    path = tmp_path / "pressure_delta.json"
    data = {
        "(0.0, 0.0, 0.0)": {"delta": 0.0},
        "(1.0, 0.0, 0.0)": {"delta": 1e-9},
        "(2.0, 0.0, 0.0)": {"delta": 0.01},
        "(3.0, 0.0, 0.0)": {"delta": 0.0001},
        "(4.0, 0.0, 0.0)": {"delta": 0.0}
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path

def test_compaction_keeps_cells_above_threshold(delta_map_path, tmp_path):
    output_path = tmp_path / "compacted.json"
    count = compact_pressure_delta_map(str(delta_map_path), str(output_path), mutation_threshold=1e-4)
    assert count == 2  # (2.0...) and (3.0...) exceed threshold
    with open(output_path, "r") as f:
        result = json.load(f)
    assert "(2.0, 0.0, 0.0)" in result
    assert "(3.0, 0.0, 0.0)" in result

def test_compaction_all_removed_below_threshold(delta_map_path, tmp_path):
    output_path = tmp_path / "compact_none.json"
    count = compact_pressure_delta_map(str(delta_map_path), str(output_path), mutation_threshold=0.1)
    assert count == 0
    assert not output_path.exists()

def test_compaction_creates_output_dir(tmp_path):
    delta_path = tmp_path / "delta.json"
    output_dir = tmp_path / "nested" / "compact"
    output_path = output_dir / "compacted.json"
    with open(delta_path, "w") as f:
        json.dump({"(0.0, 0.0, 0.0)": {"delta": 0.02}}, f)
    count = compact_pressure_delta_map(str(delta_path), str(output_path))
    assert count == 1
    assert output_path.exists()

def test_invalid_file_graceful_failure(tmp_path):
    bad_path = tmp_path / "corrupt.json"
    bad_path.write_text("not valid json")
    out_path = tmp_path / "output.json"
    result = compact_pressure_delta_map(str(bad_path), str(out_path))
    assert result == 0
    assert not out_path.exists()

def test_empty_input_file(tmp_path):
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("{}")
    out_path = tmp_path / "compact.json"
    result = compact_pressure_delta_map(str(empty_path), str(out_path))
    assert result == 0
    assert not out_path.exists()



