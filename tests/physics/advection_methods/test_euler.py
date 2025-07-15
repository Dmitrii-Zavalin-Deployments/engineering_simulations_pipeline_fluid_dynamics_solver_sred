# tests/test_euler_advection.py
# 🧪 Unit tests for Euler advection method — ensures velocity updates via upstream neighbors and fallback logic

import pytest
from src.grid_modules.cell import Cell
from src.physics.advection_methods.euler import compute_euler_velocity

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=1.0, fluid_mask=fluid)

@pytest.fixture
def base_domain():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 1, "nz": 1
        }
    }

def test_no_upstream_neighbor_keeps_velocity(base_domain):
    grid = [make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])]
    result = compute_euler_velocity(grid, dt=0.1, config=base_domain)
    assert result[0].velocity == [1.0, 0.0, 0.0]

def test_upstream_neighbor_applies_euler_update(base_domain):
    dx = 1.0  # (3.0 - 0.0)/3
    cell = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    upstream = make_cell(0.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    grid = [upstream, cell]
    result = compute_euler_velocity(grid, dt=0.1, config=base_domain)
    expected = [1.0 + 0.1 * (2.0 - 1.0) / dx, 0.0, 0.0]
    assert result[1].velocity == pytest.approx(expected)

def test_upstream_solid_cell_skipped(base_domain):
    cell = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    upstream = make_cell(0.0, 0.0, 0.0, [2.0, 0.0, 0.0], fluid=False)
    grid = [upstream, cell]
    result = compute_euler_velocity(grid, dt=0.1, config=base_domain)
    assert result[1].velocity == [1.0, 0.0, 0.0]

def test_malformed_velocity_skips_update(base_domain):
    cell = make_cell(1.0, 0.0, 0.0, "not_a_vector")
    upstream = make_cell(0.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    grid = [upstream, cell]
    result = compute_euler_velocity(grid, dt=0.1, config=base_domain)
    assert result[1].velocity == "not_a_vector"

def test_non_fluid_cell_is_unchanged(base_domain):
    cell = make_cell(1.0, 0.0, 0.0, [3.0, 0.0, 0.0], fluid=False)
    result = compute_euler_velocity([cell], dt=0.1, config=base_domain)
    assert result[0].velocity == [3.0, 0.0, 0.0]

def test_multiple_cells_updated_correctly(base_domain):
    dx = 1.0
    dt = 0.2
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    ]
    result = compute_euler_velocity(grid, dt=dt, config=base_domain)
    # Middle cell updates from first: v = 1 + 0.2*(0 - 1)/1 = 0.8
    assert result[1].velocity == pytest.approx([0.8, 0.0, 0.0])
    # Last cell updates from second: v = 2 + 0.2*(1 - 2)/1 = 1.8
    assert result[2].velocity == pytest.approx([1.8, 0.0, 0.0])

def test_zero_resolution_falls_back_to_dx_1():
    config = {"domain_definition": {"min_x": 0.0, "max_x": 1.0, "nx": 0}}
    cell = make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    upstream = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    grid = [upstream, cell]
    result = compute_euler_velocity(grid, dt=0.1, config=config)
    # dx fallback = 1.0 → expected = 0 + 0.1*(1.0 - 0.0)/1.0 = 0.1
    assert result[1].velocity == pytest.approx([0.1, 0.0, 0.0])