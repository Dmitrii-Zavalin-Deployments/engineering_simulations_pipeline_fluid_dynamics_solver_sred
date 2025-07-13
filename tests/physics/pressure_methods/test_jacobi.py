# tests/physics/pressure_methods/test_jacobi.py
# 🧪 Unit tests for Jacobi pressure solver

import pytest
from src.physics.pressure_methods.jacobi import solve_jacobi_pressure
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, pressure=0.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def make_config(nx=3, ny=1, nz=1, max_iterations=100, tolerance=1e-6):
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": nx * 1.0,
            "min_y": 0.0, "max_y": ny * 1.0,
            "min_z": 0.0, "max_z": nz * 1.0,
            "nx": nx, "ny": ny, "nz": nz
        },
        "pressure_solver": {
            "method": "jacobi",
            "max_iterations": max_iterations,
            "tolerance": tolerance
        }
    }

# ------------------------------
# solve_jacobi_pressure tests
# ------------------------------

def test_jacobi_returns_expected_length():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0]),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0])
    ]
    divergence = [1.0, -1.0]
    config = make_config()
    pressures = solve_jacobi_pressure(grid, divergence, config)
    assert isinstance(pressures, list)
    assert len(pressures) == 2
    assert all(isinstance(p, float) for p in pressures)

def test_jacobi_converges_for_uniform_input():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], pressure=10.0),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=10.0)
    ]
    divergence = [0.0, 0.0]
    config = make_config(max_iterations=20, tolerance=1e-8)
    pressures = solve_jacobi_pressure(grid, divergence, config)
    assert all(abs(p - 10.0) < 1e-2 for p in pressures)

def test_jacobi_includes_initial_pressure():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], pressure=5.0),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=0.0)
    ]
    divergence = [0.0, 0.0]
    config = make_config()
    pressures = solve_jacobi_pressure(grid, divergence, config)
    assert abs(pressures[0] - 5.0) < 1.0  # initial pressure preserved

def test_jacobi_skips_solid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=1.0)
    ]
    divergence = [0.0]
    config = make_config()
    pressures = solve_jacobi_pressure(grid, divergence, config)
    assert len(pressures) == 1
    assert isinstance(pressures[0], float)

def test_jacobi_raises_on_mismatched_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0]),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0])
    ]
    divergence = [1.0]  # should be length 2
    config = make_config()
    with pytest.raises(ValueError):
        solve_jacobi_pressure(grid, divergence, config)

def test_jacobi_handles_boundary_conditions():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0]),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0]),
        make_cell(2.0, 0.0, 0.0, [0, 0, 0])
    ]
    divergence = [1.0, 0.0, -1.0]
    config = make_config(nx=3)
    pressures = solve_jacobi_pressure(grid, divergence, config)
    assert len(pressures) == 3
    assert any(abs(p) > 0.01 for p in pressures)



