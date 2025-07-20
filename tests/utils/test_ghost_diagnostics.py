# ✅ Unit Test Suite — Ghost Diagnostics
# 📄 Full Path: tests/utils/test_ghost_diagnostics.py

import pytest
from src.utils import ghost_diagnostics as gd

class DummyGhost:
    def __init__(self, x, y, z, ghost_face=None, velocity=None, pressure=None,
                 was_enforced=False, originated_from_boundary=False):
        self.x = x
        self.y = y
        self.z = z
        self.ghost_face = ghost_face
        self.velocity = velocity
        self.pressure = pressure
        self.was_enforced = was_enforced
        self.originated_from_boundary = originated_from_boundary

class DummyFluid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.fluid_mask = True
        self.influenced_by_ghost = False

def test_dict_registry_basic_analysis():
    registry = {
        "g1": {"coordinate": (0.0, 0.0, 0.0), "face": "x+", "pressure": 1.0, "velocity": [0.0, 0.0, 0.0]},
        "g2": {"coordinate": (1.0, 0.0, 0.0), "face": "x-", "pressure": None, "velocity": [1.0, 0.0, 0.0]}
    }
    result = gd.analyze_ghost_registry(registry)
    assert result["total"] == 2
    assert result["pressure_overrides"] == 1
    assert result["no_slip_enforced"] == 1
    assert result["per_face"]["x+"] == 1
    assert result["per_face"]["x-"] == 1

def test_set_registry_basic_analysis():
    ghosts = {
        DummyGhost(0.0, 0.0, 0.0, ghost_face="y+", velocity=[0.0, 0.0, 0.0], pressure=5.0),
        DummyGhost(0.0, 1.0, 0.0, ghost_face="y-", velocity=[0.0, 1.0, 0.0], pressure=None)
    }
    result = gd.analyze_ghost_registry(ghosts)
    assert result["total"] == 2
    assert result["pressure_overrides"] == 1
    assert result["no_slip_enforced"] == 1
    assert result["per_face"]["y+"] == 1
    assert result["per_face"]["y-"] == 1

def test_fluid_adjacency_detection(monkeypatch):
    ghosts = {
        DummyGhost(1.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0])
    }
    fluid_cells = [DummyFluid(0.0, 0.0, 0.0)]
    result = gd.analyze_ghost_registry(ghosts, grid=fluid_cells)
    assert result["fluid_cells_adjacent_to_ghosts"] == 1
    assert fluid_cells[0].influenced_by_ghost is False  # no enforcement flags set

def test_fluid_adjacency_with_influence_tagging():
    ghost_meta = DummyGhost(1.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0],
                            was_enforced=True, originated_from_boundary=False)
    fluid = DummyFluid(0.0, 0.0, 0.0)
    registry = { "g": {"coordinate": (1.0, 0.0, 0.0), "face": "x+", "pressure": 1.0,
                       "velocity": [0.0, 0.0, 0.0], "was_enforced": True } }
    result = gd.analyze_ghost_registry(registry, grid=[fluid])
    assert fluid.influenced_by_ghost is True
    assert result["fluid_cells_adjacent_to_ghosts"] == 1

def test_type_error_on_invalid_registry_type():
    with pytest.raises(TypeError):
        gd.analyze_ghost_registry(["not", "a", "valid", "ghost", "set"])

def test_log_ghost_summary_output(capsys):
    registry = {
        "g1": {"coordinate": (1.0, 0.0, 0.0), "face": "x+", "pressure": 1.0, "velocity": [0.0, 0.0, 0.0]}
    }
    fluid = DummyFluid(0.0, 0.0, 0.0)
    gd.log_ghost_summary(registry, grid=[fluid])
    output = capsys.readouterr().out
    assert "Ghost Cells:" in output
    assert "Ghost Pressure Overrides:" in output
    assert "No-slip Velocity Enforced:" in output
    assert "Fluid cells bordering ghosts:" in output

def test_inject_diagnostics_attaches_field_and_logs(capsys):
    registry = {
        "g1": {"coordinate": (1.0, 0.0, 0.0), "face": "x-", "pressure": 1.0, "velocity": [0.0, 0.0, 0.0]}
    }
    fluid = DummyFluid(0.0, 0.0, 0.0)
    snap = {"step_index": 1}
    updated = gd.inject_diagnostics(snap, registry, grid=[fluid])
    output = capsys.readouterr().out
    assert "Ghost Cells:" in output
    assert "ghost_diagnostics" in updated
    assert updated["ghost_diagnostics"]["total"] == 1



