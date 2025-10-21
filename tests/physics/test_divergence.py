import pytest
from src.physics.divergence import compute_divergence
from src.physics.divergence_methods.divergence_helpers import get_neighbor_velocity, central_difference
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
        velocity=velocity,
        pressure=133.0,
        fluid_mask=fluid_mask
    )

# 🧪 Test: Basic divergence computation
def test_divergence_basic():
    grid = [fluid_cell(velocity=[1.0, 0.0, 0.0])]
    config = minimal_config()
    result = compute_divergence(grid, config, debug=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], float)

# 🧪 Test: Ghost cell exclusion
def test_divergence_excludes_ghosts(capfd):
    ghost = fluid_cell(velocity=[1.0, 0.0, 0.0])
    grid = [ghost]
    ghost_registry = {id(ghost)}
    result = compute_divergence(grid, minimal_config(), ghost_registry, debug=True)
    out, _ = capfd.readouterr()
    assert "[DEBUG] ⛔️ Skipping ghost cell" in out
    assert result == []

# 🧪 Test: Malformed velocity is downgraded
def test_divergence_downgrades_malformed_velocity(capfd):
    bad_cell = fluid_cell(velocity=None, fluid_mask=True)
    grid = [bad_cell]
    result = compute_divergence(grid, minimal_config(), debug=True)
    out, _ = capfd.readouterr()
    assert "[DEBUG] ⚠️ Downgrading cell[0]" in out or "[DEBUG] ⚠️ Downgrading cell" in out
    assert result == []

# 🧪 Test: Non-fluid cell is excluded
def test_divergence_excludes_non_fluid(capfd):
    grid = [fluid_cell(velocity=[1.0, 0.0, 0.0], fluid_mask=False)]
    result = compute_divergence(grid, minimal_config(), debug=True)
    out, _ = capfd.readouterr()
    assert "[DEBUG] ⚠️ Downgrading cell[0]" in out or "[DEBUG] ⚠️ Downgrading cell" in out
    assert result == []

# 🧪 Test: Multiple fluid cells with mixed validity
def test_divergence_mixed_cells(capfd):
    valid = fluid_cell(x=0.0, velocity=[1.0, 0.0, 0.0])
    ghost = fluid_cell(x=1.0, velocity=[1.0, 0.0, 0.0])
    malformed = fluid_cell(x=2.0, velocity=None, fluid_mask=True)
    ghost_registry = {id(ghost)}
    grid = [valid, ghost, malformed]
    result = compute_divergence(grid, minimal_config(), ghost_registry, debug=True)
    out, _ = capfd.readouterr()
    assert "[DEBUG] ✅ Safe grid assembled" in out
    assert "[DEBUG] ⚠️ Downgrading cell[2]" in out or "[DEBUG] ⚠️ Downgrading cell" in out
    assert len(result) == 1
    assert isinstance(result[0], float)

# 🧪 Test: Verbose logging output
def test_divergence_verbose_logging(capfd):
    grid = [fluid_cell(velocity=[1.0, 0.0, 0.0])]
    compute_divergence(grid, minimal_config(), verbose=True, debug=False)
    out, _ = capfd.readouterr()
    assert "🧭 Divergence at" in out
    assert "📈 Max divergence" in out

# 🧪 Test: Debug + Verbose together
def test_divergence_debug_and_verbose(capfd):
    grid = [fluid_cell(velocity=[1.0, 0.0, 0.0])]
    compute_divergence(grid, minimal_config(), verbose=True, debug=True)
    out, _ = capfd.readouterr()
    assert "[DEBUG] ✅ Safe grid assembled" in out
    assert "🧭 Divergence at" in out
    assert "📈 Max divergence" in out

# 🧪 Covers line 40: valid neighbor with velocity
def test_get_neighbor_velocity_valid():
    grid_lookup = {
        (0.0, 0.0, 0.0): Cell(
            x=0.0, y=0.0, z=0.0,
            velocity=[1.0, 2.0, 3.0],
            pressure=101.0,
            fluid_mask=True
        )
    }
    result = get_neighbor_velocity(grid_lookup, 0.0, 0.0, 0.0, 'x', 0, 1.0)
    assert result == [1.0, 2.0, 3.0]

# 🧪 Covers line 62: both neighbors present
def test_central_difference_valid():
    v_pos = [2.0, 4.0, 6.0]
    v_neg = [1.0, 3.0, 5.0]
    spacing = 0.5
    component = 1  # y-direction
    result = central_difference(v_pos, v_neg, spacing, component)
    expected = (4.0 - 3.0) / (2.0 * 0.5)
    assert result == expected



