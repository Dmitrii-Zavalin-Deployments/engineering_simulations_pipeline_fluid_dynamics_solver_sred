# tests/utils/test_ghost_diagnostics.py
# ✅ Unit tests for ghost_diagnostics.py — verifies analysis, logging, and injection behavior

import pytest
from src.utils.ghost_diagnostics import analyze_ghost_registry, inject_diagnostics

class MockGhostCell:
    def __init__(self, x, y, z, face="x_min", pressure=20.0, velocity=[0.0, 0.0, 0.0]):
        self.x = x
        self.y = y
        self.z = z
        self.face = face         # ✅ used by analyze_ghost_registry
        self.ghost_face = face   # preserves original attribute
        self.pressure = pressure
        self.velocity = velocity

# 🔧 Sample dict-style ghost registry fixture
@pytest.fixture
def sample_dict_registry():
    return {
        101: {"face": "x_min", "origin": (0, 1, 1), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]},
        102: {"face": "x_max", "origin": (2, 1, 1), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]},
        103: {"face": "y_min", "origin": (1, 0, 1), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]},
        104: {"face": "y_min", "origin": (1, 0, 2), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]},
        105: {"face": "z_max", "origin": (1, 1, 2), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]},
        106: {"face": "z_max", "origin": (2, 1, 2), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]},
    }

# 🔧 Sample set-style ghost registry fixture
@pytest.fixture
def sample_set_registry():
    return {
        MockGhostCell(0.0, 1.0, 1.0, face="x_min"),
        MockGhostCell(2.0, 1.0, 1.0, face="x_max"),
        MockGhostCell(1.0, 0.0, 1.0, face="y_min"),
        MockGhostCell(1.0, 0.0, 2.0, face="y_min"),
        MockGhostCell(1.0, 1.0, 2.0, face="z_max"),
        MockGhostCell(2.0, 1.0, 2.0, face="z_max"),
    }

def test_analyze_dict_registry_totals(sample_dict_registry):
    summary = analyze_ghost_registry(sample_dict_registry)
    assert summary["total"] == 6
    assert isinstance(summary["per_face"], dict)

def test_analyze_dict_registry_per_face_counts(sample_dict_registry):
    summary = analyze_ghost_registry(sample_dict_registry)
    assert summary["per_face"] == {
        "x_min": 1,
        "x_max": 1,
        "y_min": 2,
        "z_max": 2
    }

def test_analyze_set_registry_totals(sample_set_registry):
    summary = analyze_ghost_registry(sample_set_registry)
    assert summary["total"] == 6
    assert isinstance(summary["per_face"], dict)

def test_analyze_set_registry_per_face_counts(sample_set_registry):
    summary = analyze_ghost_registry(sample_set_registry)
    assert summary["per_face"] == {
        "x_min": 1,
        "x_max": 1,
        "y_min": 2,
        "z_max": 2
    }

def test_registry_dict_with_missing_face_key():
    registry = {
        201: {"origin": (1, 0, 0)},
        202: {"face": "y_max", "origin": (1, 2, 0), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]}
    }
    summary = analyze_ghost_registry(registry)
    assert summary["total"] == 2
    assert summary["per_face"] == {"y_max": 1}

def test_registry_dict_with_none_face():
    registry = {
        301: {"face": None, "origin": (0, 1, 0), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]},
        302: {"face": "x_max", "origin": (2, 1, 0), "pressure": 20.0, "velocity": [0.0, 0.0, 0.0]}
    }
    summary = analyze_ghost_registry(registry)
    assert summary["total"] == 2
    assert summary["per_face"] == {"x_max": 1}

def test_inject_diagnostics_into_snapshot_with_dict(sample_dict_registry):
    snapshot = {"step_index": 0, "grid": []}
    updated = inject_diagnostics(snapshot, sample_dict_registry)
    assert "ghost_diagnostics" in updated
    assert updated["ghost_diagnostics"]["total"] == 6
    assert updated["ghost_diagnostics"]["per_face"]["y_min"] == 2

def test_inject_diagnostics_into_snapshot_with_set(sample_set_registry):
    snapshot = {"step_index": 0, "grid": []}
    updated = inject_diagnostics(snapshot, sample_set_registry)
    assert "ghost_diagnostics" in updated
    assert updated["ghost_diagnostics"]["total"] == 6
    assert updated["ghost_diagnostics"]["per_face"]["y_min"] == 2



