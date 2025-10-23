import pytest
from src.physics.divergence_methods.divergence_helpers import get_neighbor_velocity, central_difference
from src.grid_modules.cell import Cell

def test_get_neighbor_velocity_valid():
    grid_lookup = {
        (0.0, 0.0, 0.0): Cell(
            x=0.0, y=0.0, z=0.0,
            velocity=[1.0, 2.0, 3.0],
            pressure=101.0,
            fluid_mask=True
        )
    }
    result = get_neighbor_velocity(grid_lookup, 0.0, 0.0, 0.0, 'x', 0, 1.0)
    assert result == [1.0, 2.0, 3.0]

def test_central_difference_valid():
    v_pos = [2.0, 4.0, 6.0]
    v_neg = [1.0, 3.0, 5.0]
    spacing = 0.5
    component = 1
    result = central_difference(v_pos, v_neg, spacing, component)
    assert result == (4.0 - 3.0) / (2.0 * 0.5)



