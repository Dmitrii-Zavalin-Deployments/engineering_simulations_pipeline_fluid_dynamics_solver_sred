# tests/physics/divergence_methods/test_central.py
# 🧪 Unit tests for compute_central_divergence — structured-grid central difference

import pytest
from src.physics.divergence_methods.central import compute_central_divergence
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

# ------------------------------
# compute_central_divergence tests
# ------------------------------

def test_uniform_velocity_1d_returns_zero_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    ]
    config = make_config()
    divergence = compute_central_divergence(grid, config)
    for val in divergence:
        assert val == pytest.approx(0.0)

def test_linear_velocity_gradient_x_direction_returns_nonzero_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, [3.0, 0.0, 0.0])
    ]
    config = make_config()
    divergence = compute_central_divergence(grid, config)
    assert len(divergence) == 3
    assert divergence[1] == pytest.approx(1.0)

def test_missing_neighbors_on_edge_returns_zero():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    ]
    config = make_config(nx=2)
    divergence = compute_central_divergence(grid, config)
    assert len(divergence) == 2
    for val in divergence:
        assert isinstance(val, float)

def test_divergence_only_computed_for_fluid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=True),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=False),
        make_cell(2.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=True)
    ]
    config = make_config()
    divergence = compute_central_divergence(grid, config)
    assert len(divergence) == 2

def test_divergence_includes_y_and_z_gradients():
    grid = [
        make_cell(1.0, 0.0, 1.0, [0.0, 1.0, 2.0]),  # center
        make_cell(1.0, 1.0, 1.0, [0.0, 2.0, 2.0]),  # y+
        make_cell(1.0, -1.0, 1.0, [0.0, 0.0, 2.0]), # y-
        make_cell(1.0, 0.0, 2.0, [0.0, 1.0, 3.0]),  # z+
        make_cell(1.0, 0.0, 0.0, [0.0, 1.0, 1.0])   # z-
    ]
    config = make_config(dx=1.0, dy=1.0, dz=1.0, nx=1, ny=3, nz=3)
    divergence = compute_central_divergence(grid, config)
    assert len(divergence) == 5
    center_index = 0
    assert divergence[center_index] == pytest.approx(1.0)  # ∂vy/∂y = 1.0, ∂vz/∂z = 1.0 → total ≈ 2.0

def test_malformed_velocity_skipped_safely():
    grid = [
        make_cell(0.0, 0.0, 0.0, "invalid"),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    ]
    config = make_config()
    divergence = compute_central_divergence(grid, config)
    assert len(divergence) == 1
    assert divergence[0] == pytest.approx(0.0)



