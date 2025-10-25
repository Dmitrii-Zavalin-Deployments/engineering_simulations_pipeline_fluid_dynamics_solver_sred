import pytest
from src.physics.ghost_influence_applier import apply_ghost_influence
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def test_influence_applied_to_adjacent_fluid_cell():
    ghost = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=5.0, fluid_mask=False)
    fluid = make_cell(0.5, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    grid = [ghost, fluid]
    count = apply_ghost_influence(grid, spacing=(0.5, 0.5, 0.5), verbose=False)
    assert count == 1
    assert fluid.velocity == [1.0, 0.0, 0.0]
    assert fluid.pressure == 5.0
    assert getattr(fluid, "influenced_by_ghost", False) is True

def test_no_influence_if_fields_match():
    ghost = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=5.0, fluid_mask=False)
    fluid = make_cell(0.5, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=5.0, fluid_mask=True)
    grid = [ghost, fluid]
    count = apply_ghost_influence(grid, spacing=(0.5, 0.5, 0.5), verbose=False)
    assert count == 0
    assert fluid.velocity == [1.0, 0.0, 0.0]
    assert fluid.pressure == 5.0
    assert not hasattr(fluid, "influenced_by_ghost")
    assert getattr(fluid, "triggered_by", None) == "ghost adjacency — no mutation (fields matched)"

def test_influence_skipped_for_non_adjacent_cells():
    ghost = make_cell(0.0, 0.0, 0.0, velocity=[2.0, 2.0, 2.0], pressure=10.0, fluid_mask=False)
    fluid = make_cell(2.0, 2.0, 2.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    grid = [ghost, fluid]
    count = apply_ghost_influence(grid, spacing=(0.5, 0.5, 0.5), verbose=False)
    assert count == 0
    assert fluid.velocity == [0.0, 0.0, 0.0]
    assert fluid.pressure == 0.0

def test_multiple_ghosts_influence_multiple_fluid_cells():
    ghost1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=5.0, fluid_mask=False)
    ghost2 = make_cell(1.0, 1.0, 1.0, velocity=[0.0, 1.0, 0.0], pressure=10.0, fluid_mask=False)
    fluid1 = make_cell(0.5, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    fluid2 = make_cell(1.5, 1.0, 1.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    grid = [ghost1, ghost2, fluid1, fluid2]
    count = apply_ghost_influence(grid, spacing=(0.5, 0.5, 0.5), verbose=False)
    assert count == 2
    assert fluid1.velocity == [1.0, 0.0, 0.0]
    assert fluid1.pressure == 5.0
    assert fluid2.velocity == [0.0, 1.0, 0.0]
    assert fluid2.pressure == 10.0

def test_influence_respects_radius_parameter():
    ghost = make_cell(0.0, 0.0, 0.0, velocity=[9.0, 9.0, 9.0], pressure=99.0, fluid_mask=False)
    fluid = make_cell(1.5, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    grid = [ghost, fluid]
    count = apply_ghost_influence(grid, spacing=(0.5, 0.5, 0.5), radius=3, verbose=False)
    assert count == 1
    assert fluid.velocity == [9.0, 9.0, 9.0]
    assert fluid.pressure == 99.0

def test_skips_solid_cells():
    ghost = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 1.0, 1.0], pressure=1.0, fluid_mask=False)
    solid = make_cell(0.5, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=False)
    grid = [ghost, solid]
    count = apply_ghost_influence(grid, spacing=(0.5, 0.5, 0.5), verbose=False)
    assert count == 0
    assert solid.velocity == [0.0, 0.0, 0.0]
    assert solid.pressure == 0.0



