# tests/physics/test_pressure_projection.py
# 🧪 Unit tests for pressure projection — validate incompressibility enforcement and ghost-aware mutation logic

import pytest
from src.physics.pressure_projection import solve_pressure_poisson
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, pressure=0.0, fluid_mask=True, ghost_face=None):
    c = Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)
    if ghost_face:
        setattr(c, "ghost_face", ghost_face)
    return c

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

def test_projection_preserves_structure_and_masks():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=True),
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

def test_pressure_changes_on_nonzero_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], pressure=1.0),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=1.0)
    ]
    divergence = [0.2, -0.2]
    config = make_config()
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    pressures = [c.pressure for c in result if c.fluid_mask]
    assert any(abs(p - 1.0) > 1e-3 for p in pressures)
    assert mutated is True

def test_projection_respects_zero_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], pressure=3.0),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=9.0)
    ]
    divergence = [0.0, 0.0]
    config = make_config()
    result, _ = solve_pressure_poisson(grid, divergence, config)
    pressures = [c.pressure for c in result if c.fluid_mask]
    midpoint = sum(pressures) / len(pressures)
    assert all(abs(p - midpoint) < 1.0 for p in pressures)

def test_skips_solid_cells_correctly():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0, 0, 0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [0, 0, 0], pressure=0.0)
    ]
    divergence = [0.1]
    config = make_config()
    result, _ = solve_pressure_poisson(grid, divergence, config)
    assert result[0].pressure is None
    assert result[1].pressure is not None

def test_empty_grid_returns_clean():
    result, mutated = solve_pressure_poisson([], [], make_config())
    assert result == []
    assert mutated is False

def test_mismatch_divergence_raises():
    grid = [make_cell(0.0, 0.0, 0.0, [0, 0, 0]), make_cell(1.0, 0.0, 0.0, [0, 0, 0])]
    with pytest.raises(ValueError):
        solve_pressure_poisson(grid, [1.0], make_config())

def test_unknown_solver_method_raises():
    grid = [make_cell(0.0, 0.0, 0.0, [0, 0, 0])]
    with pytest.raises(ValueError):
        solve_pressure_poisson(grid, [0.0], make_config(method="not_supported"))

def test_handles_bad_velocity_gracefully():
    bad_grid = [make_cell(0.0, 0.0, 0.0, "bad", pressure=2.0)]
    with pytest.raises(ValueError):
        solve_pressure_poisson(bad_grid, [], make_config())

def test_converges_with_high_iterations():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], pressure=2.0),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], pressure=4.0)
    ]
    divergence = [0.1, -0.1]
    config = make_config(iterations=200, tolerance=1e-8)
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    pressures = [c.pressure for c in result if c.fluid_mask]
    assert len(pressures) == 2
    assert any(abs(p - o) > 1e-3 for p, o in zip(pressures, [2.0, 4.0]))
    assert mutated is True

def test_projection_respects_ghost_pressure_boundaries():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.5, 0.0, 0.0], pressure=0.0),
        make_cell(1.0, 0.0, 0.0, [0.5, 0.0, 0.0], pressure=0.0),
        make_cell(2.0, 0.0, 0.0, [0.5, 0.0, 0.0], pressure=20.0, fluid_mask=False, ghost_face="x_max")
    ]
    divergence = [0.1, 0.1]
    config = make_config()
    result, mutated = solve_pressure_poisson(grid, divergence, config)
    pressures = [c.pressure for c in result[:2]]
    assert any(abs(p - 0.0) > 1e-2 for p in pressures), "❌ Fluid pressure unchanged despite ghost pressure enforcement"
    assert mutated is True



