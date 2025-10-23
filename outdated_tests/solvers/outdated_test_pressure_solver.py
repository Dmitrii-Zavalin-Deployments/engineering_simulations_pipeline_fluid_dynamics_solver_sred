import pytest
import os
import json
from src.solvers.pressure_solver import apply_pressure_correction
from src.grid_modules.cell import Cell
from src.exporters.velocity_field_writer import write_velocity_field
import src.physics.divergence as divergence_module

# 🔧 Enable centralized debug for divergence module
@pytest.fixture(autouse=True)
def enable_debug(monkeypatch):
    monkeypatch.setattr(divergence_module, "DEBUG", True)

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
    cell2.boundary_type = "outlet"
    cell3 = Cell(x=0.3, y=0.3, z=0.3, velocity=[0.0, 0.0, 1.0], pressure=1.0, fluid_mask=True)
    cell3.influenced_by_ghost = True
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
    _, _, _, metadata = result
    assert metadata["pressure_mutation_count"] == 0

# ✅ Test: Ghost-influenced cells are tagged
def test_ghost_influence_tagging(mock_grid, mock_input_data):
    result = apply_pressure_correction(mock_grid, mock_input_data, step=3)
    _, _, _, metadata = result
    ghost_registry = metadata.get("ghost_registry", {})
    assert isinstance(ghost_registry, dict)
    assert any("coordinate" in entry for entry in ghost_registry.values())

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

# ✅ Test: Mutation logic and ghost tagging triggered
def test_pressure_mutation_triggered():
    # Create three adjacent fluid cells with a velocity gradient
    cell1 = Cell(x=0.5, y=0.5, z=0.5, velocity=[100.0, 0.0, 0.0], pressure=-1.0, fluid_mask=True)
    cell2 = Cell(x=0.6, y=0.5, z=0.5, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    cell3 = Cell(x=0.4, y=0.5, z=0.5, velocity=[50.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    cell1.influenced_by_ghost = True
    grid = [cell1, cell2, cell3]

    input_data = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 10, "ny": 10, "nz": 10
        },
        "simulation_parameters": {"time_step": 0.05},
        "grid_resolution": "normal",
        "ghost_trigger_chain": [],
        "solver_parameters": {
            "max_pressure_iterations": 50,
            "pressure_tolerance": 1e-5
        }
    }

    grid_out, _, _, metadata = apply_pressure_correction(grid, input_data, step=6)
    assert metadata["pressure_mutation_count"] >= 1
    assert any("mutation_triggered_by" in vars(c) and c.mutation_triggered_by == "ghost_influence" for c in grid_out)

# ✅ Test: Velocity field export integrity
def test_velocity_field_export(tmp_path):
    cell = Cell(x=0.5, y=0.5, z=0.5, velocity=[100.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    grid = [cell]
    step = 7
    output_dir = tmp_path / "snapshots"
    output_dir.mkdir()

    write_velocity_field(grid, step, output_dir=str(output_dir))

    expected_path = output_dir / f"velocity_field_step_{step:04d}.json"
    assert expected_path.exists()
    print(f"[TEST] Velocity field file created at: {expected_path}")

    with open(expected_path, "r") as f:
        data = json.load(f)
        key = "(0.50, 0.50, 0.50)"
        assert key in data
        assert data[key]["vx"] == 100.0
        assert data[key]["vy"] == 0.0
        assert data[key]["vz"] == 0.0



