# tests/physics/test_divergence.py
# 🧪 Validates central-difference divergence with ghost exclusion and safety filtering

import pytest
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(
        x=x, y=y, z=z,
        velocity=velocity,
        pressure=0.0,
        fluid_mask=fluid
    )

@pytest.fixture
def config_3x1x1():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }

def test_basic_divergence_computation(config_3x1x1):
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    ]
    result = compute_divergence(grid, config=config_3x1x1)
    assert result[1] == pytest.approx(1.0)

def test_skips_ghost_cells(config_3x1x1):
    ghost = make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    fluid = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    grid = [ghost, fluid]
    ghost_registry = {id(ghost)}
    result = compute_divergence(grid, config=config_3x1x1, ghost_registry=ghost_registry)
    assert len(result) == 0 or result[0] == pytest.approx(0.0)

def test_invalid_velocity_marked_as_solid(config_3x1x1):
    bad = Cell(x=1.0, y=0.0, z=0.0, velocity="invalid", pressure=5.0, fluid_mask=True)
    result = compute_divergence([bad], config=config_3x1x1)
    assert result == []

def test_velocity_none_excluded(config_3x1x1):
    broken = Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=1.0, fluid_mask=True)
    result = compute_divergence([broken], config=config_3x1x1)
    assert result == []

def test_solid_cell_filtered(config_3x1x1):
    solid = make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0], fluid=False)
    result = compute_divergence([solid], config=config_3x1x1)
    assert result == []

def test_verbose_logging_output(capsys, config_3x1x1):
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    ]
    compute_divergence(grid, config=config_3x1x1, verbose=True)
    out = capsys.readouterr().out
    assert "🧭 Divergence at" in out
    assert "📈 Max divergence" in out

def test_empty_grid_returns_empty_list(config_3x1x1):
    result = compute_divergence([], config=config_3x1x1)
    assert result == []

def test_mixed_cell_types_and_exclusion(config_3x1x1):
    ghost = make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    fluid = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    solid = make_cell(2.0, 0.0, 0.0, [2.0, 0.0, 0.0], fluid=False)
    bad = Cell(x=3.0, y=0.0, z=0.0, velocity=None, pressure=1.0, fluid_mask=True)
    grid = [ghost, fluid, solid, bad]
    ghost_registry = {id(ghost)}
    result = compute_divergence(grid, config=config_3x1x1, ghost_registry=ghost_registry)
    assert len(result) == 1
    assert result[0] == pytest.approx(0.0)



