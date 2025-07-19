# tests/physics/test_ghost_influence_applier.py
# 🧪 Unit tests for src/physics/ghost_influence_applier.py

from src.grid_modules.cell import Cell
from src.physics.ghost_influence_applier import apply_ghost_influence

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_ghost_influence_applied_to_velocity_and_pressure():
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    ghost = make_cell(1.0, 1.0, 1.0, velocity=[2.0, 3.0, 4.0], pressure=5.0, fluid=False)
    result = apply_ghost_influence([fluid, ghost], spacing=(1.0, 1.0, 1.0))
    assert result == 1
    assert fluid.velocity == [2.0, 3.0, 4.0]
    assert fluid.pressure == 5.0
    assert getattr(fluid, "influenced_by_ghost", False)

def test_ghost_influence_skipped_when_fields_match():
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[2.2, 3.3, 4.4], pressure=5.5)
    ghost = make_cell(1.0, 1.0, 1.0, velocity=[2.2, 3.3, 4.4], pressure=5.5, fluid=False)
    result = apply_ghost_influence([fluid, ghost], spacing=(1.0, 1.0, 1.0))
    assert result == 0
    assert getattr(fluid, "ghost_influence_attempted", True)
    assert not getattr(fluid, "influenced_by_ghost", False)
    assert fluid.velocity == [2.2, 3.3, 4.4]
    assert fluid.pressure == 5.5

def test_ghost_influence_only_applied_to_adjacent_cells():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    ghost = make_cell(3.0, 3.0, 3.0, velocity=[9.0, 9.0, 9.0], pressure=9.0, fluid=False)
    result = apply_ghost_influence([fluid, ghost], spacing=(1.0, 1.0, 1.0))
    assert result == 0
    assert not getattr(fluid, "influenced_by_ghost", False)
    assert fluid.velocity == [0.0, 0.0, 0.0]
    assert fluid.pressure == 0.0

def test_multiple_adjacent_fluid_cells_influenced():
    fluid1 = make_cell(1.0, 1.0, 1.0, velocity=[0.0, 0.0, 0.0], pressure=1.0)
    fluid2 = make_cell(1.0, 2.0, 1.0, velocity=[0.0, 0.0, 0.0], pressure=2.0)
    ghost = make_cell(1.0, 1.5, 1.0, velocity=[5.0, 5.0, 5.0], pressure=5.5, fluid=False)
    result = apply_ghost_influence([fluid1, fluid2, ghost], spacing=(1.0, 1.0, 1.0))
    assert result == 2
    for fluid in [fluid1, fluid2]:
        assert fluid.velocity == [5.0, 5.0, 5.0]
        assert fluid.pressure == 5.5

def test_influence_radius_extends_neighbor_distance():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    ghost = make_cell(2.0, 2.0, 2.0, velocity=[9.9, 9.9, 9.9], pressure=9.9, fluid=False)
    result = apply_ghost_influence([fluid, ghost], spacing=(1.0, 1.0, 1.0), radius=2)
    assert result == 1
    assert fluid.velocity == [9.9, 9.9, 9.9]
    assert fluid.pressure == 9.9



