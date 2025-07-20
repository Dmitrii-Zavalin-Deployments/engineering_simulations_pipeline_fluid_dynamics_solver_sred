# ✅ Unit Test Suite — Output Manager
# 📄 Full Path: tests/test_output_manager.py

import pytest
from dataclasses import dataclass
from src.output_manager import validate_snapshot, serialize_snapshot, EXPECTED_SNAPSHOT_KEYS

@dataclass
class DummyCell:
    x: int
    y: int
    z: int
    velocity: float
    pressure: float
    fluid_mask: bool

def test_validate_snapshot_with_all_keys():
    snapshot = {key: True for key in EXPECTED_SNAPSHOT_KEYS}
    assert validate_snapshot(snapshot) is True

def test_validate_snapshot_missing_keys():
    snapshot = {key: True for key in EXPECTED_SNAPSHOT_KEYS[:-1]}  # one key removed
    assert validate_snapshot(snapshot) is False

def test_serialize_snapshot_converts_cells():
    snapshot = {"step_index": 0}
    grid = [
        DummyCell(x=0, y=0, z=0, velocity=1.0, pressure=0.1, fluid_mask=True),
        DummyCell(x=1, y=1, z=1, velocity=2.0, pressure=0.2, fluid_mask=False)
    ]
    result = serialize_snapshot(snapshot.copy(), grid)
    assert "grid" in result
    assert isinstance(result["grid"], list)
    assert result["grid"][0]["x"] == 0
    assert result["grid"][1]["fluid_mask"] is False

def test_serialize_snapshot_preserves_existing_keys():
    snapshot = {"step_index": 42, "global_cfl": 0.5}
    grid = [DummyCell(x=0, y=0, z=0, velocity=0.0, pressure=0.0, fluid_mask=True)]
    result = serialize_snapshot(snapshot.copy(), grid)
    assert result["step_index"] == 42
    assert result["global_cfl"] == 0.5
    assert len(result["grid"]) == 1



