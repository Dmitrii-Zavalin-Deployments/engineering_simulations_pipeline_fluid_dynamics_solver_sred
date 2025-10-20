import pytest
from src.physics.divergence import compute_divergence
from src.grid_modules.cell import Cell

def minimal_config():
    return {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 1, "ny": 1, "nz": 1
    }

def fluid_cell(x=0.0, y=0.0, z=0.0, velocity=None, fluid_mask=True):
    return Cell(
        x=x, y=y, z=z,
        velocity=velocity if velocity else [1.0, 0.0, 0.0],
        pressure=133.0,
        fluid_mask=fluid_mask
    )

# 🧪 Test: Basic divergence computation
def test_divergence_basic():
    grid = [fluid_cell()]
    config = minimal_config()
    result = compute_divergence(grid, config)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], float)

# 🧪 Test: Ghost cell exclusion
def test_divergence_excludes_ghosts():
    ghost = fluid_cell()
    grid = [ghost]
    ghost_registry = {id(ghost)}
    result = compute_divergence(grid, minimal_config(), ghost_registry)
    assert result == []

# 🧪 Test: Malformed velocity is downgraded
def test_divergence_downgrades_malformed_velocity():
    bad_cell = fluid_cell(velocity=None)
    grid = [bad_cell]
    result = compute_divergence(grid, minimal_config())
    assert result == []

# 🧪 Test: Non-fluid cell is excluded
def test_divergence_excludes_non_fluid():
    grid = [fluid_cell(fluid_mask=False)]
    result = compute_divergence(grid, minimal_config())
    assert result == []

# 🧪 Test: Multiple fluid cells with mixed validity
def test_divergence_mixed_cells():
    valid = fluid_cell(x=0.0)
    ghost = fluid_cell(x=1.0)
    malformed = fluid_cell(x=2.0, velocity=None)
    ghost_registry = {id(ghost)}
    grid = [valid, ghost, malformed]
    result = compute_divergence(grid, minimal_config(), ghost_registry)
    assert len(result) == 1
    assert isinstance(result[0], float)

# 🧪 Test: Verbose logging output
def test_divergence_verbose_logging(capfd):
    grid = [fluid_cell()]
    compute_divergence(grid, minimal_config(), verbose=True)
    out, _ = capfd.readouterr()
    assert "🧭 Divergence at" in out
    assert "📈 Max divergence" in out



