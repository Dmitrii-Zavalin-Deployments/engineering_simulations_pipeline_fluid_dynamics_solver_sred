# tests/test_velocity_projection.py
# 🧪 Unit tests for pressure-based velocity projection logic

import pytest
from src.grid_modules.cell import Cell
from src.physics.velocity_projection import apply_pressure_velocity_projection

@pytest.fixture
def config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 1, "nz": 1
        }
    }

@pytest.fixture
def grid_1d(config):
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    return [
        Cell(x=0.5*dx, y=0.5, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=10.0, fluid_mask=True),
        Cell(x=1.5*dx, y=0.5, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=20.0, fluid_mask=True),
        Cell(x=2.5*dx, y=0.5, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=30.0, fluid_mask=True)
    ]

def test_velocity_projection_applies_gradients(grid_1d, config):
    projected = apply_pressure_velocity_projection(grid_1d, config)
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    expected_grad_x = (30.0 - 10.0) / (2.0 * dx)
    expected_velocity = [1.0 - expected_grad_x, 0.0, 0.0]
    assert projected[1].velocity == pytest.approx(expected_velocity, abs=1e-6)

def test_edge_cells_keep_original_velocity(grid_1d, config):
    projected = apply_pressure_velocity_projection(grid_1d, config)
    assert projected[0].velocity == [1.0, 0.0, 0.0]
    assert projected[2].velocity == [1.0, 0.0, 0.0]

def test_non_fluid_cells_are_unchanged(config):
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    grid = [
        Cell(x=0.5*dx, y=0.5, z=0.5, velocity=None, pressure=None, fluid_mask=False),
        Cell(x=1.5*dx, y=0.5, z=0.5, velocity=None, pressure=50.0, fluid_mask=False),
        Cell(x=2.5*dx, y=0.5, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=100.0, fluid_mask=True)
    ]
    result = apply_pressure_velocity_projection(grid, config)
    assert result[0].velocity is None
    assert result[1].velocity is None
    assert isinstance(result[2].velocity, list)

def test_projection_skips_if_no_neighbors(config):
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    single = Cell(x=1.5*dx, y=0.5, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=100.0, fluid_mask=True)
    result = apply_pressure_velocity_projection([single], config)
    assert result[0].velocity == [1.0, 0.0, 0.0]

def test_projection_applies_multi_axis_gradient(config):
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    dy = (config["domain_definition"]["max_y"] - config["domain_definition"]["min_y"]) / config["domain_definition"]["ny"]

    center = Cell(x=1.5*dx, y=0.5, z=0.5, velocity=[1.0, 1.0, 0.0], pressure=20.0, fluid_mask=True)
    left = Cell(x=0.5*dx, y=0.5, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=10.0, fluid_mask=True)
    right = Cell(x=2.5*dx, y=0.5, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=30.0, fluid_mask=True)
    down = Cell(x=1.5*dx, y=0.0, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=10.0, fluid_mask=True)
    up = Cell(x=1.5*dx, y=1.0, z=0.5, velocity=[1.0, 0.0, 0.0], pressure=30.0, fluid_mask=True)

    grid = [left, right, down, up, center]
    result = apply_pressure_velocity_projection(grid, config)

    grad_x = (30.0 - 10.0) / (2.0 * dx)

    # ✅ Only X gradient modifies velocity — Y pressure gradient not applied to Y velocity component
    expected_velocity = [1.0 - grad_x, 1.0, 0.0]
    assert result[-1].velocity == pytest.approx(expected_velocity, abs=1e-6)



