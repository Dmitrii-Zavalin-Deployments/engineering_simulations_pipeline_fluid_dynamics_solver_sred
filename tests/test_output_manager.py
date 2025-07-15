# tests/test_output_manager.py
# 🧪 Validates snapshot schema integrity and Cell serialization logic

from src.output_manager import validate_snapshot, serialize_snapshot, EXPECTED_SNAPSHOT_KEYS
from src.grid_modules.cell import Cell
from dataclasses import dataclass

def test_validate_snapshot_with_all_keys():
    snapshot = {key: True for key in EXPECTED_SNAPSHOT_KEYS}
    assert validate_snapshot(snapshot) is True

def test_validate_snapshot_missing_one_key():
    keys = EXPECTED_SNAPSHOT_KEYS[:-1]  # remove last
    snapshot = {key: True for key in keys}
    assert validate_snapshot(snapshot) is False

def test_validate_snapshot_empty_dict_fails():
    assert validate_snapshot({}) is False

def test_serialize_snapshot_converts_cell_objects():
    snapshot = {
        "step_index": 0,
        "max_velocity": 1.0,
        "max_divergence": 0.5,
        "global_cfl": 0.9,
        "overflow_detected": False,
        "damping_enabled": False,
        "projection_passes": 2
    }

    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0], pressure=5.0, fluid_mask=True),
        Cell(x=1.0, y=0.0, z=0.0, velocity=[0.0, 1.0, 0.0], pressure=4.5, fluid_mask=False)
    ]

    result = serialize_snapshot(snapshot.copy(), grid)
    assert "grid" in result
    assert isinstance(result["grid"], list)
    assert isinstance(result["grid"][0], dict)
    assert result["grid"][0]["x"] == 0.0
    assert result["grid"][1]["fluid_mask"] is False

def test_serialize_snapshot_grid_overwrites_existing_grid_key():
    snapshot = {key: True for key in EXPECTED_SNAPSHOT_KEYS}
    snapshot["grid"] = "placeholder"
    grid = [Cell(x=0.0, y=0.0, z=0.0, velocity=[0.0]*3, pressure=0.0, fluid_mask=True)]
    result = serialize_snapshot(snapshot, grid)
    assert isinstance(result["grid"], list)
    assert isinstance(result["grid"][0], dict)

def test_serialize_snapshot_with_empty_grid():
    snapshot = {key: True for key in EXPECTED_SNAPSHOT_KEYS}
    result = serialize_snapshot(snapshot.copy(), [])
    assert "grid" in result
    assert isinstance(result["grid"], list)
    assert result["grid"] == []

def test_serialize_snapshot_preserves_other_fields():
    snapshot = {
        "step_index": 3,
        "max_velocity": 0.0,
        "max_divergence": 0.0,
        "global_cfl": 0.1,
        "overflow_detected": False,
        "damping_enabled": True,
        "projection_passes": 0
    }
    grid = [Cell(x=1, y=2, z=3, velocity=[0.0]*3, pressure=1.0, fluid_mask=True)]
    result = serialize_snapshot(snapshot.copy(), grid)
    assert result["step_index"] == 3
    assert result["damping_enabled"] is True
    assert result["grid"][0]["pressure"] == 1.0