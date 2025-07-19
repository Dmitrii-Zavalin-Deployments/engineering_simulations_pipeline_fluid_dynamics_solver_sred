# tests/physics/advection_methods/test_euler.py
# 🧪 Unit tests for src/physics/advection_methods/euler.py

from src.grid_modules.cell import Cell
from src.physics.advection_methods.euler import compute_euler_velocity

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=1.0, fluid_mask=fluid)

def test_returns_original_velocity_for_non_fluid_cells():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid=False)
    result = compute_euler_velocity([c1], dt=0.1, config={})
    assert result[0].velocity == [1.0, 0.0, 0.0]

def test_returns_original_velocity_for_invalid_velocity_field():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=None)
    result = compute_euler_velocity([c1], dt=0.1, config={})
    assert result[0].velocity is None

def test_skips_update_if_no_upstream_neighbor():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    config = {"domain_definition": {"nx": 10, "min_x": 0.0, "max_x": 1.0}}
    result = compute_euler_velocity([c1], dt=0.1, config=config)
    assert result[0].velocity == [1.0, 0.0, 0.0]

def test_applies_euler_update_with_upstream_neighbor():
    # cell at x=0.1, neighbor at x=0.0
    neighbor = make_cell(0.0, 0.0, 0.0, velocity=[2.0, 0.0, 0.0])
    target = make_cell(0.1, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    grid = [neighbor, target]
    config = {"domain_definition": {"nx": 10, "min_x": 0.0, "max_x": 1.0}}
    dt = 0.1

    result = compute_euler_velocity(grid, dt=dt, config=config)
    updated = result[1]

    dx = (1.0 - 0.0) / 10
    expected = [1.0 + (dt / dx) * (2.0 - 1.0), 0.0, 0.0]
    assert updated.velocity == expected

def test_uses_fallback_if_upstream_is_non_fluid():
    neighbor = make_cell(0.0, 0.0, 0.0, velocity=[9.0, 0.0, 0.0], fluid=False)
    target = make_cell(0.1, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    grid = [neighbor, target]
    config = {"domain_definition": {"nx": 10, "min_x": 0.0, "max_x": 1.0}}
    result = compute_euler_velocity(grid, dt=0.1, config=config)
    assert result[1].velocity == [1.0, 0.0, 0.0]



