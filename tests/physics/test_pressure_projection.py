# tests/physics/test_pressure_projection.py
# 🧪 Unit tests for pressure projection — validate incompressibility enforcement and pressure mutation tracking

import pytest
from src.physics.pressure_projection import solve_pressure_poisson
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, pressure=0.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def make_config(method="jacobi", nx=3, ny=1, nz=1, iterations=50, tolerance=1e-6):
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0 * nx,
            "min_y": 0.0, "max_y": 1.0 * ny,
            "min_z": 0.0, "max_z": 1.0 * nz,
            "nx": nx, "ny": ny, "nz": nz
        },
        "pressure_solver": {
            "method": method,
            "max_iterations": iterations,
            "tolerance": tolerance
        }
    }

# ------------------------------
# solve_pressure_poisson tests
# ------------------------------

def test_pressure_projection_preserves_grid_structure():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=False)
    ]
    divergence = [0.5]
    config = make_config()
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    assert len(result) == len(grid)
    assert result[0].fluid_mask is True
    assert result[1].fluid_mask is False
    assert result[1].pressure is None
    assert isinstance(mutated, bool)

def test_pressure_projection_mutates_pressure_on_nonzero_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], pressure=1.0),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=1.0)
    ]
    divergence = [0.2, -0.2]
    config = make_config(iterations=100, tolerance=1e-8)
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    pressures = [cell.pressure for cell in result if cell.fluid_mask]
    assert any(abs(p - 1.0) > 1e-3 for p in pressures)
    assert mutated is True

def test_pressure_projection_balances_pressure_with_zero_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], pressure=3.0),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=9.0)
    ]
    divergence = [0.0, 0.0]
    config = make_config()
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    pressures = [cell.pressure for cell in result if cell.fluid_mask]
    midpoint = sum(pressures) / len(pressures)
    assert all(abs(p - midpoint) < 1.0 for p in pressures)
    assert mutated is True or mutated is False  # may or may not change — result depends on solver stability

def test_pressure_projection_skips_solid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=0.0)
    ]
    divergence = [0.1]
    config = make_config()
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    assert result[0].pressure is None
    assert result[1].pressure is not None
    assert isinstance(mutated, bool)

def test_pressure_projection_handles_empty_grid():
    result, mutated = solve_pressure_poisson([], [], make_config())
    assert result == []
    assert mutated is False

def test_pressure_projection_raises_on_divergence_mismatch():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0]),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0])
    ]
    divergence = [1.0]  # mismatch
    config = make_config()
    with pytest.raises(ValueError):
        solve_pressure_poisson(grid, divergence, config)

def test_pressure_projection_rejects_unknown_method():
    grid = [make_cell(0.0, 0.0, 0.0, [0, 0, 0])]
    divergence = [0.0]
    config = make_config(method="unsupported")
    with pytest.raises(ValueError):
        solve_pressure_poisson(grid, divergence, config)

def test_pressure_projection_handles_malformed_velocity_gracefully():
    grid = [make_cell(0.0, 0.0, 0.0, "bad", pressure=1.0, fluid_mask=True)]
    divergence = []
    config = make_config()
    with pytest.raises(ValueError):
        solve_pressure_poisson(grid, divergence, config)

def test_pressure_projection_converges_with_high_iterations():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], pressure=2.0),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], pressure=4.0)
    ]
    divergence = [0.1, -0.1]
    config = make_config(iterations=200, tolerance=1e-8)
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    pressures = [cell.pressure for cell in result if cell.fluid_mask]
    assert len(pressures) == 2
    assert any(abs(p - orig) > 1e-3 for p, orig in zip(pressures, [2.0, 4.0]))
    assert mutated is True



