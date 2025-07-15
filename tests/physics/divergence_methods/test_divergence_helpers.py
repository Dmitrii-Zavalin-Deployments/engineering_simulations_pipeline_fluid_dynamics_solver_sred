# tests/physics/divergence_methods/test_divergence_helpers.py
# 🧪 Unit tests for divergence_helpers — validates neighbor velocity extraction and central gradient logic

from src.grid_modules.cell import Cell
from src.physics.divergence_methods.divergence_helpers import (
    get_neighbor_velocity,
    central_gradient
)

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=1.0, fluid_mask=fluid)

def test_get_neighbor_velocity_returns_valid_vector():
    neighbor = make_cell(1.0, 0.0, 0.0, [1.0, 2.0, 3.0])
    lookup = {(1.0, 0.0, 0.0): neighbor}
    result = get_neighbor_velocity(lookup, x=0.0, y=0.0, z=0.0, axis='x', sign=+1, spacing=1.0)
    assert result == [1.0, 2.0, 3.0]

def test_get_neighbor_velocity_skips_solid_cell():
    neighbor = make_cell(1.0, 0.0, 0.0, [3.0, 3.0, 3.0], fluid=False)
    lookup = {(1.0, 0.0, 0.0): neighbor}
    result = get_neighbor_velocity(lookup, x=0.0, y=0.0, z=0.0, axis='x', sign=+1, spacing=1.0)
    assert result is None

def test_get_neighbor_velocity_skips_invalid_velocity():
    neighbor = make_cell(1.0, 0.0, 0.0, "invalid_vector")
    lookup = {(1.0, 0.0, 0.0): neighbor}
    result = get_neighbor_velocity(lookup, x=0.0, y=0.0, z=0.0, axis='x', sign=+1, spacing=1.0)
    assert result is None

def test_get_neighbor_velocity_negative_direction():
    neighbor = make_cell(-1.0, 0.0, 0.0, [0.0, 1.0, 2.0])
    lookup = {(-1.0, 0.0, 0.0): neighbor}
    result = get_neighbor_velocity(lookup, x=0.0, y=0.0, z=0.0, axis='x', sign=-1, spacing=1.0)
    assert result == [0.0, 1.0, 2.0]

def test_get_neighbor_velocity_y_and_z_axes():
    y_neighbor = make_cell(0.0, 2.0, 0.0, [1.0, 2.0, 3.0])
    z_neighbor = make_cell(0.0, 0.0, -2.0, [3.0, 2.0, 1.0])
    lookup = {(0.0, 2.0, 0.0): y_neighbor, (0.0, 0.0, -2.0): z_neighbor}
    vy = get_neighbor_velocity(lookup, 0.0, 1.0, 0.0, 'y', +1, 1.0)
    vz = get_neighbor_velocity(lookup, 0.0, 0.0, -1.0, 'z', -1, 1.0)
    assert vy == [1.0, 2.0, 3.0]
    assert vz == [3.0, 2.0, 1.0]

def test_central_gradient_returns_expected_value():
    vp = [2.0, 4.0, 6.0]
    vm = [0.0, 2.0, 4.0]
    result = central_gradient(vp, vm, spacing=1.0, component=0)
    assert result == pytest.approx(1.0)  # (2 - 0)/2

def test_central_gradient_zero_if_missing_neighbors():
    vp = None
    vm = [1.0, 2.0, 3.0]
    result = central_gradient(vp, vm, spacing=1.0, component=0)
    assert result == 0.0

    vp = [4.0, 5.0, 6.0]
    vm = None
    result = central_gradient(vp, vm, spacing=1.0, component=1)
    assert result == 0.0

def test_central_gradient_handles_vector_component_index():
    vp = [10.0, 20.0, 30.0]
    vm = [8.0, 16.0, 24.0]
    dx = 2.0
    for i, expected in enumerate([(10 - 8)/4, (20 - 16)/4, (30 - 24)/4]):
        result = central_gradient(vp, vm, dx, component=i)
        assert result == pytest.approx(expected)