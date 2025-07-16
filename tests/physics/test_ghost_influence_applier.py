# tests/physics/test_ghost_influence_applier.py
# 🧪 Validates ghost-to-fluid influence mechanics across adjacency, mutation triggers, and tagging logic

import pytest
from src.grid_modules.cell import Cell
from src.physics.ghost_influence_applier import apply_ghost_influence

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

@pytest.fixture
def spacing():
    return (1.0, 1.0, 1.0)

def test_influence_applied_to_adjacent_fluid(spacing):
    fluid = make_cell(1.0, 1.0, 1.0, [0.0, 0.0, 0.0], 0.0)
    ghost = make_cell(2.0, 1.0, 1.0, [1.0, 0.0, 0.0], 5.0, fluid=False)
    grid = [fluid, ghost]
    count = apply_ghost_influence(grid, spacing)
    assert count == 1
    assert fluid.velocity == [1.0, 0.0, 0.0]
    assert fluid.pressure == 5.0
    assert getattr(fluid, "influenced_by_ghost", False)

def test_no_influence_if_values_identical(spacing):
    fluid = make_cell(1.0, 1.0, 1.0, [1.0, 1.0, 1.0], 5.0)
    ghost = make_cell(2.0, 1.0, 1.0, [1.0, 1.0, 1.0], 5.0, fluid=False)
    count = apply_ghost_influence([fluid, ghost], spacing)
    assert count == 0
    assert not hasattr(fluid, "influenced_by_ghost")

def test_influence_respects_radius(spacing):
    fluid = make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], 0.0)
    ghost = make_cell(3.0, 0.0, 0.0, [1.0, 0.0, 0.0], 9.9, fluid=False)
    count = apply_ghost_influence([fluid, ghost], spacing, radius=2)
    assert count == 0  # Too far for radius=2
    count = apply_ghost_influence([fluid, ghost], spacing, radius=3)
    assert count == 1

def test_multiple_ghosts_influence_multiple_fluid_cells(spacing):
    f1 = make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], 0.0)
    f2 = make_cell(1.0, 1.0, 1.0, [0.0, 0.0, 0.0], 0.0)
    g1 = make_cell(0.0, 1.0, 0.0, [2.0, 2.0, 2.0], 2.0, fluid=False)
    g2 = make_cell(2.0, 1.0, 1.0, [3.0, 3.0, 3.0], 3.0, fluid=False)
    grid = [f1, f2, g1, g2]
    count = apply_ghost_influence(grid, spacing)
    assert count == 3
    assert f1.velocity == [2.0, 2.0, 2.0]
    assert f2.velocity == [3.0, 3.0, 3.0]

def test_influence_skips_malformed_velocity(spacing):
    fluid = make_cell(1.0, 1.0, 1.0, [0.0, 0.0, 0.0], 0.0)
    ghost = make_cell(2.0, 1.0, 1.0, "invalid", 10.0, fluid=False)
    grid = [fluid, ghost]
    count = apply_ghost_influence(grid, spacing)
    assert count == 0
    assert fluid.velocity == [0.0, 0.0, 0.0]  # unchanged
    assert fluid.pressure == 0.0  # ✅ unchanged due to invalid ghost velocity

def test_influence_skips_non_numeric_pressure(spacing):
    fluid = make_cell(1.0, 1.0, 1.0, [0.0, 0.0, 0.0], 0.0)
    ghost = make_cell(2.0, 1.0, 1.0, [1.0, 0.0, 0.0], "bad", fluid=False)
    count = apply_ghost_influence([fluid, ghost], spacing)
    assert count == 1
    assert fluid.velocity == [1.0, 0.0, 0.0]
    assert fluid.pressure == 0.0  # unchanged

def test_verbose_logging_output(capsys, spacing):
    fluid = make_cell(1.0, 1.0, 1.0, [0.0, 0.0, 0.0], 0.0)
    ghost = make_cell(2.0, 1.0, 1.0, [9.9, 9.9, 9.9], 9.9, fluid=False)
    apply_ghost_influence([fluid, ghost], spacing, verbose=True)
    out = capsys.readouterr().out
    assert "Ghost" in out and "influenced fluid" in out

def test_ghosts_do_not_affect_non_fluid_cells(spacing):
    solid = make_cell(1.0, 1.0, 1.0, [0.0, 0.0, 0.0], 0.0, fluid=False)
    ghost = make_cell(2.0, 1.0, 1.0, [1.0, 0.0, 0.0], 5.0, fluid=False)
    count = apply_ghost_influence([solid, ghost], spacing)
    assert count == 0
    assert not hasattr(solid, "influenced_by_ghost")

def test_rounding_tolerance_applied_correctly():
    fluid = make_cell(1.0000001, 1.0, 1.0, [0.0, 0.0, 0.0], 0.0)
    ghost = make_cell(2.0, 1.0, 1.0, [1.0, 0.0, 0.0], 5.0, fluid=False)
    spacing = (1.0, 1.0, 1.0)
    count = apply_ghost_influence([fluid, ghost], spacing)
    assert count == 1



