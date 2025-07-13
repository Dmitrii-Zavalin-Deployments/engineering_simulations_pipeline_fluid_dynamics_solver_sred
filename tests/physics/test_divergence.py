# tests/physics/test_divergence.py
# 🧪 Unit tests for compute_divergence — central-difference velocity divergence

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

def test_uniform_velocity_zero_divergence():
    # 1D grid of 3 cells with uniform velocity → divergence should be 0
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    ]
    config = make_config()
    result = compute_divergence(grid, config)
    for value in result:
        assert pytest.approx(value) == 0.0

def test_linear_velocity_gradient_x_direction():
    # 1D grid with increasing vx → divergence should be positive
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [3.0, 0.0, 0.0])
    ]
    config = make_config()
    result = compute_divergence(grid, config)

    # Expect non-zero divergence at center (only fluid cell with full neighbors)
    assert len(result) == 3
    assert pytest.approx(result[1]) > 0.0

def test_divergence_excludes_solid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=True),
        make_cell(2.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=False)
    ]
    config = make_config()
    result = compute_divergence(grid, config)
    assert len(result) == 1
    assert pytest.approx(result[0]) == 0.0

def test_missing_neighbor_edge_case_handled_safely():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0])
        # No neighbor at x = 2.0 for central difference
    ]
    config = make_config(nx=2)
    result = compute_divergence(grid, config)
    assert len(result) == 2
    for value in result:
        assert isinstance(value, float)

def test_divergence_returns_ordered_values_for_fluid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=True),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=False),
        make_cell(2.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=True)
    ]
    config = make_config()
    result = compute_divergence(grid, config)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)

def test_divergence_safety_for_malformed_velocity_vector():
    bad_cell = make_cell(0.0, 0.0, 0.0, "not_a_vector")
    config = make_config()
    try:
        result = compute_divergence([bad_cell], config)
        assert isinstance(result[0], float)
    except Exception:
        pytest.fail("compute_divergence crashed on malformed velocity input")



