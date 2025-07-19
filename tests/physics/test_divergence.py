# tests/physics/test_divergence.py
# 🧪 Unit tests for src/physics/divergence.py

from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=1.0, fluid_mask=fluid)

def test_returns_empty_for_non_fluid_cells():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid=False)
    c2 = make_cell(1.0, 0.0, 0.0, velocity=None, fluid=True)
    result = compute_divergence([c1, c2])
    assert result == []

def test_excludes_ghost_cells_from_computation():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    c2 = make_cell(1.0, 0.0, 0.0, velocity=[2.0, 0.0, 0.0])
    ghost_registry = {id(c2)}
    result = compute_divergence([c1, c2], ghost_registry=ghost_registry)
    assert len(result) == 1

def test_computes_valid_divergence_for_fluid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[2.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, velocity=[3.0, 0.0, 0.0])
    ]
    config = {
        "domain_definition": {
            "nx": 2, "min_x": 0.0, "max_x": 2.0,
            "ny": 1, "min_y": 0.0, "max_y": 1.0,
            "nz": 1, "min_z": 0.0, "max_z": 1.0
        }
    }
    result = compute_divergence(grid, config=config)
    assert len(result) == 3
    assert any(abs(val) > 0.0 for val in result)

def test_malformed_velocity_excluded_from_computation():
    good = make_cell(1.0, 0.0, 0.0, velocity=[2.0, 0.0, 0.0])
    malformed = make_cell(0.0, 0.0, 0.0, velocity=None)
    result = compute_divergence([malformed, good])
    assert len(result) == 1

def test_verbose_logging_runs_without_crashing(capsys):
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    c2 = make_cell(1.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    config = {
        "domain_definition": {
            "nx": 1, "min_x": 0.0, "max_x": 1.0,
            "ny": 1, "min_y": 0.0, "max_y": 1.0,
            "nz": 1, "min_z": 0.0, "max_z": 1.0
        }
    }
    compute_divergence([c1, c2], config=config, verbose=True)
    output = capsys.readouterr().out
    assert "Divergence at" in output
    assert "Max divergence" in output or "returned empty list" in output



