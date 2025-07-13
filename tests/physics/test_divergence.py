# tests/physics/test_divergence.py
# 🧪 Unit tests for compute_divergence — central-difference velocity divergence and ghost-awareness

import pytest
from src.physics.divergence import compute_divergence
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, fluid_mask=True, pressure=1.0):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def make_config(dx=1.0, dy=1.0, dz=1.0, nx=3, ny=1, nz=1):
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": dx * nx,
            "min_y": 0.0, "max_y": dy * ny,
            "min_z": 0.0, "max_z": dz * nz,
            "nx": nx, "ny": ny, "nz": nz
        }
    }

# -------------------------------
# Divergence Value Test Scenarios
# -------------------------------

def test_uniform_velocity_yields_zero_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    ]
    config = make_config()
    result = compute_divergence(grid, config)
    assert all(pytest.approx(val, abs=1e-6) == 0.0 for val in result)

def test_linear_x_velocity_gives_correct_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [3.0, 0.0, 0.0])
    ]
    config = make_config()
    result = compute_divergence(grid, config)
    assert len(result) == 3
    assert result[1] == pytest.approx(1.0, abs=1e-6)

def test_solid_cells_excluded_from_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0], fluid_mask=True),
        make_cell(2.0, 0.0, 0.0, [3.0, 0.0, 0.0], fluid_mask=False)
    ]
    config = make_config()
    result = compute_divergence(grid, config)
    assert len(result) == 1
    assert isinstance(result[0], float)

def test_edge_cells_divergence_handles_missing_neighbors():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 1.0, 1.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 2.0, 2.0])
    ]
    config = make_config(nx=2)
    result = compute_divergence(grid, config)
    assert len(result) == 2
    assert all(isinstance(val, float) for val in result)

def test_divergence_order_matches_fluid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.1, 0.0, 0.0], fluid_mask=True),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=False),
        make_cell(2.0, 0.0, 0.0, [0.2, 0.0, 0.0], fluid_mask=True)
    ]
    config = make_config()
    result = compute_divergence(grid, config)
    assert len(result) == 2
    assert all(isinstance(val, float) for val in result)

def test_malformed_velocity_excluded_from_divergence():
    bad = make_cell(0.0, 0.0, 0.0, "bad_vector", fluid_mask=True)
    config = make_config()
    result = compute_divergence([bad], config)
    assert isinstance(result, list)
    assert len(result) == 0

def test_divergence_skips_malformed_velocity_among_valid():
    grid = [
        make_cell(0.0, 0.0, 0.0, "bad"),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, "bad"),
        make_cell(3.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    ]
    config = make_config(nx=4)
    result = compute_divergence(grid, config)
    assert len(result) == 2
    assert all(isinstance(val, float) for val in result)

def test_divergence_excludes_ghost_cells():
    ghost = make_cell(-1.0, 0.0, 0.0, None, fluid_mask=False)
    fluid = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    grid = [ghost, fluid]
    config = make_config()
    ghost_registry = {id(ghost)}
    result = compute_divergence(grid, config, ghost_registry=ghost_registry)
    assert len(result) == 1
    assert isinstance(result[0], float)



