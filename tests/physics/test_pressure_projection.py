# tests/physics/test_pressure_projection.py
# 🧪 Unit tests for src/physics/pressure_projection.py

from src.grid_modules.cell import Cell
from src.physics.pressure_projection import solve_pressure_poisson, extract_ghost_coords

def make_cell(x, y, z, velocity=None, pressure=None, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_extract_ghost_coords_identifies_ghosts():
    g1 = make_cell(0.0, 0.0, 0.0, fluid=False)
    g2 = make_cell(1.0, 0.0, 0.0, fluid=False)
    setattr(g1, "ghost_face", "x_min")
    setattr(g2, "ghost_face", "x_max")
    fluid = make_cell(0.5, 0.5, 0.5)
    result = extract_ghost_coords([fluid, g1, g2])
    assert (0.0, 0.0, 0.0) in result
    assert (1.0, 0.0, 0.0) in result
    assert (0.5, 0.5, 0.5) not in result

def test_pressure_poisson_raises_on_divergence_length_mismatch():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0)
    config = {"pressure_solver": {"method": "jacobi"}}
    try:
        solve_pressure_poisson([fluid], [], config)
        assert False  # Should raise
    except ValueError as e:
        assert "does not match" in str(e)

def test_pressure_poisson_applies_projection_and_updates_pressure():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=1.0)
    config = {
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "pressure_solver": {
            "method": "jacobi", "max_iterations": 5, "tolerance": 1e-6
        }
    }
    result, mutated = solve_pressure_poisson([fluid], divergence=[1.0], config=config)
    assert isinstance(result, list)
    assert isinstance(result[0].pressure, float)
    assert mutated is True

def test_pressure_poisson_skips_ghost_cell_during_projection():
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[0.0, 0.0, 0.0], pressure=2.0)
    ghost = make_cell(0.0, 1.0, 1.0, fluid=False)
    setattr(ghost, "ghost_face", "x_min")
    config = {
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "pressure_solver": {
            "method": "jacobi", "max_iterations": 4
        }
    }
    result, mutated = solve_pressure_poisson([fluid, ghost], divergence=[0.0], config=config)
    assert len(result) == 2
    assert result[1].fluid_mask is False
    assert result[1].pressure is None



