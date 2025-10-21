import pytest
from src.solvers.pressure_solver import apply_pressure_correction
from src.grid_modules.cell import Cell

# 🔧 Mock dependencies
@pytest.fixture
def mock_input_data():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 10, "ny": 10, "nz": 10
        },
        "simulation_parameters": {
            "time_step": 0.05
        },
        "grid_resolution": "normal",
        "ghost_trigger_chain": []
    }

@pytest.fixture
def mock_grid():
    cell1 = Cell(x=0.1, y=0.1, z=0.1, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)

    cell2 = Cell(x=0.2, y=0.2, z=0.2, velocity=[0.0, 1.0, 0.0], pressure=1.0, fluid_mask=True)
    cell2.boundary_type = "outlet"  # ✅ Set manually

    cell3 = Cell(x=0.3, y=0.3, z=0.3, velocity=[0.0, 0.0, 1.0], pressure=1.0, fluid_mask=True)
    cell3.influenced_by_ghost = True  # ✅ Set manually

    cell4 = Cell(x=0.4, y=0.4, z=0.4, velocity=None, pressure=None, fluid_mask=False)

    return [cell1, cell2, cell3, cell4]

# ✅ Test: Basic pressure correction flow
def test_pressure_correction_runs(mock_grid, mock_input_data):
    result = apply_pressure_correction(mock_grid, mock_input_data, step=1)
    assert isinstance(result, tuple)
    grid_out, mutated, passes, metadata = result
    assert isinstance(grid_out, list)
    assert isinstance(mutated, bool)
    assert isinstance(passes, int)
    assert isinstance(metadata, dict)
    assert "max_divergence" in metadata
    assert "pressure_mutation_count" in metadata
    assert "ghost_registry" in metadata

# ✅ Test: Neumann enforcement skips outlet/wall cells
def test_neumann_boundary_skipped(mock_grid, mock_input_data):
    result = apply_pressure_correction(mock_grid, mock_input_data, step=2)
    grid_out, _, _, metadata = result
    outlet_cells = [c for c in grid_out if getattr(c, "boundary_type", None) == "outlet"]
    assert outlet_cells, "No outlet cell found in grid output"
    assert not getattr(outlet_cells[0], "pressure_mutated", False)

# ✅ Test: Ghost-influenced cells are tagged
def test_ghost_influence_tagging(mock_grid, mock_input_data):
    result = apply_pressure_correction(mock_grid, mock_input_data, step=3)
    grid_out, _, _, metadata = result
    ghost_cells = [c for c in grid_out if getattr(c, "influenced_by_ghost", False)]
    assert ghost_cells, "No ghost-influenced cell found in grid output"
    assert getattr(ghost_cells[0], "mutation_triggered_by", None) == "ghost_influence"

# ✅ Test: No fluid cells → no mutation
def test_empty_fluid_grid():
    empty_cell = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False)
    empty_grid = [empty_cell]
    input_data = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 10, "ny": 10, "nz": 10
        },
        "simulation_parameters": {"time_step": 0.05},
        "grid_resolution": "normal"
    }
    result = apply_pressure_correction(empty_grid, input_data, step=4)
    _, _, _, metadata = result
    assert metadata["pressure_mutation_count"] == 0

# ✅ Test: Config validation failure
def test_invalid_config_rejected():
    bad_config = {"domain_definition": {"min_x": 0}}  # Missing required keys
    with pytest.raises(Exception):
        apply_pressure_correction([], bad_config, step=5)



