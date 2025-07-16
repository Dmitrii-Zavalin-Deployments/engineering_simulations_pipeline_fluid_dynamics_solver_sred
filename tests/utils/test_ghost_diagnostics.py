# tests/utils/test_ghost_diagnostics.py
# 🧪 Validates ghost cell analysis, summary logging, and snapshot injection

import pytest
from src.grid_modules.cell import Cell
from src.utils import ghost_diagnostics

def make_ghost_meta(face, coord, pressure=None, velocity=None):
    return {
        "face": face,
        "coordinate": coord,
        "pressure": pressure,
        "velocity": velocity
    }

def make_ghost_cell(x, y, z, face=None, pressure=None, velocity=None):
    c = Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=False)
    if face:
        setattr(c, "ghost_face", face)
    return c

def make_fluid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=5.0, fluid_mask=True)

def test_analyze_ghost_registry_dict_mode_counts_faces():
    registry = {
        1: make_ghost_meta("x_min", (0.0, 0.0, 0.0)),
        2: make_ghost_meta("x_max", (2.0, 0.0, 0.0)),
        3: make_ghost_meta("x_min", (0.0, 1.0, 0.0))
    }
    summary = ghost_diagnostics.analyze_ghost_registry(registry)
    assert summary["total"] == 3
    assert summary["per_face"]["x_min"] == 2
    assert summary["per_face"]["x_max"] == 1

def test_analyze_ghost_registry_list_mode_pressure_and_velocity():
    ghost1 = make_ghost_cell(0.0, 0.0, 0.0, face="x_min", pressure=99.0, velocity=[0.0, 0.0, 0.0])
    ghost2 = make_ghost_cell(1.0, 0.0, 0.0, face="x_max", pressure=101.0, velocity=[1.0, 0.0, 0.0])
    ghost3 = make_ghost_cell(2.0, 0.0, 0.0, face="x_max", pressure=None, velocity=[0.0, 0.0, 0.0])
    registry = [ghost1, ghost2, ghost3]
    summary = ghost_diagnostics.analyze_ghost_registry(registry)
    assert summary["pressure_overrides"] == 2
    assert summary["no_slip_enforced"] == 2
    assert summary["per_face"]["x_min"] == 1
    assert summary["per_face"]["x_max"] == 2

def test_analyze_ghost_registry_fluid_adjacency():
    ghost = make_ghost_cell(1.0, 1.0, 1.0)
    fluid1 = make_fluid_cell(1.0, 2.0, 1.0)  # y+1
    fluid2 = make_fluid_cell(4.0, 4.0, 4.0)  # not adjacent
    registry = [ghost]
    grid = [fluid1, fluid2]
    summary = ghost_diagnostics.analyze_ghost_registry(registry, grid=grid, spacing=(1.0, 1.0, 1.0))
    assert summary["fluid_cells_adjacent_to_ghosts"] == 1

def test_analyze_ghost_registry_empty_dict():
    summary = ghost_diagnostics.analyze_ghost_registry({})
    assert summary["total"] == 0
    assert summary["pressure_overrides"] == 0
    assert summary["no_slip_enforced"] == 0
    assert summary["fluid_cells_adjacent_to_ghosts"] == 0

def test_analyze_ghost_registry_raises_on_invalid_type():
    with pytest.raises(TypeError, match="ghost_registry must be dict or set"):
        ghost_diagnostics.analyze_ghost_registry(ghost_registry=[1, 2, 3])

def test_log_ghost_summary_prints_summary(capsys):
    ghost = make_ghost_cell(0.0, 0.0, 0.0, face="x_min", pressure=99.0, velocity=[0.0, 0.0, 0.0])
    fluid = make_fluid_cell(0.0, 1.0, 0.0)
    registry = [ghost]
    ghost_diagnostics.log_ghost_summary(registry, grid=[fluid], spacing=(1.0, 1.0, 1.0))
    out = capsys.readouterr().out
    assert "Ghost Cells" in out
    assert "x_min" in out
    assert "Pressure Overrides" in out
    assert "No-slip Velocity Enforced" in out
    assert "bordering ghosts" in out

def test_inject_diagnostics_attaches_to_snapshot(capsys):
    ghost = make_ghost_cell(0.0, 0.0, 0.0, face="y_max", pressure=55.0, velocity=[0.0, 0.0, 0.0])
    fluid = make_fluid_cell(0.0, -1.0, 0.0)
    registry = [ghost]
    snapshot = {"step_index": 0}
    updated = ghost_diagnostics.inject_diagnostics(snapshot, registry, grid=[fluid], spacing=(1.0, 1.0, 1.0))
    assert "ghost_diagnostics" in updated
    assert updated["ghost_diagnostics"]["pressure_overrides"] == 1
    out = capsys.readouterr().out
    assert "Ghost Cells" in out and "y_max" in out