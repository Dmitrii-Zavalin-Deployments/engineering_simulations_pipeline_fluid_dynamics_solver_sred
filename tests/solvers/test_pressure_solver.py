import pathlib
import pytest
from unittest.mock import patch
from src.solvers.pressure_solver import apply_pressure_correction
from src.grid_modules.cell import Cell

def make_cell(x, y, z, pressure=0.0, velocity=None, fluid_mask=True, boundary_type=None, influenced_by_ghost=False):
    cell = Cell(x=x, y=y, z=z, pressure=pressure, velocity=velocity or [0.0, 0.0, 0.0], fluid_mask=fluid_mask)
    cell.boundary_type = boundary_type
    cell.influenced_by_ghost = influenced_by_ghost
    return cell

@pytest.fixture
def base_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 1.0,
            "output_interval": 1
        },
        "ghost_trigger_chain": [0]
    }

@pytest.fixture(autouse=True)
def ensure_output_dir_exists(tmp_path):
    # Use tmp_path to avoid filesystem errors in CI
    return tmp_path

@patch("src.solvers.pressure_solver.validate_config")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
@patch("src.solvers.pressure_solver.solve_pressure_poisson")
@patch("src.solvers.pressure_solver.get_delta_threshold")
@patch("src.solvers.pressure_solver.log_reflex_pathway")
@patch("src.solvers.pressure_solver.export_pressure_delta_map")
@patch("src.solvers.pressure_solver.run_verification_if_triggered")
def test_pressure_mutation_tagging_and_metadata(
    mock_verifier, mock_export, mock_log, mock_threshold, mock_poisson, mock_div_stats, mock_validate,
    base_config, ensure_output_dir_exists
):
    cell = make_cell(0.0, 0.0, 0.0, pressure=1.0)
    updated_cell = make_cell(0.0, 0.0, 0.0, pressure=1.2)
    mock_div_stats.return_value = {"divergence": [0.05], "max": 0.05}
    mock_poisson.return_value = ([updated_cell], True, {"ghost": "registry"})
    mock_threshold.return_value = 0.1

    result_grid, pressure_mutated, passes, metadata = apply_pressure_correction(
        [cell], base_config, step=1, output_folder=str(ensure_output_dir_exists)
    )

    assert pressure_mutated is True
    assert passes == 1
    assert metadata["pressure_mutation_count"] == 1
    assert metadata["ghost_registry"] == {"ghost": "registry"}
    assert getattr(result_grid[0], "pressure_mutated", False) is True
    assert result_grid[0].mutation_source == "pressure_solver"
    assert result_grid[0].mutation_step == 1
    mock_log.assert_called_once()
    mock_export.assert_called_once()
    mock_verifier.assert_called_once()

@patch("src.solvers.pressure_solver.validate_config")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
@patch("src.solvers.pressure_solver.solve_pressure_poisson")
@patch("src.solvers.pressure_solver.get_delta_threshold")
@patch("src.solvers.pressure_solver.log_reflex_pathway")
@patch("src.solvers.pressure_solver.export_pressure_delta_map")
@patch("src.solvers.pressure_solver.run_verification_if_triggered")
def test_pressure_solver_skips_outlet_and_wall_cells(
    mock_verifier, mock_export, mock_log, mock_threshold, mock_poisson, mock_div_stats, mock_validate,
    base_config, ensure_output_dir_exists
):
    wall_cell = make_cell(0.0, 0.0, 0.0, pressure=1.0, boundary_type="wall")
    outlet_cell = make_cell(1.0, 0.0, 0.0, pressure=1.0, boundary_type="outlet")
    fluid_cell = make_cell(0.0, 1.0, 0.0, pressure=1.0)

    updated_wall = make_cell(0.0, 0.0, 0.0, pressure=1.2, boundary_type="wall")
    updated_outlet = make_cell(1.0, 0.0, 0.0, pressure=1.2, boundary_type="outlet")
    updated_fluid = make_cell(0.0, 1.0, 0.0, pressure=1.2)

    mock_div_stats.return_value = {"divergence": [0.01, 0.01, 0.01], "max": 0.01}
    mock_poisson.return_value = ([updated_wall, updated_outlet, updated_fluid], True, {})
    mock_threshold.return_value = 0.1

    result_grid, pressure_mutated, passes, metadata = apply_pressure_correction(
        [wall_cell, outlet_cell, fluid_cell], base_config, step=2, output_folder=str(ensure_output_dir_exists)
    )

    assert metadata["pressure_mutation_count"] == 1
    assert getattr(result_grid[2], "pressure_mutated", False) is True
    assert getattr(result_grid[0], "pressure_mutated", False) is False
    assert getattr(result_grid[1], "pressure_mutated", False) is False

@patch("src.solvers.pressure_solver.validate_config")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
@patch("src.solvers.pressure_solver.solve_pressure_poisson")
@patch("src.solvers.pressure_solver.get_delta_threshold")
@patch("src.solvers.pressure_solver.log_reflex_pathway")
@patch("src.solvers.pressure_solver.export_pressure_delta_map")
@patch("src.solvers.pressure_solver.run_verification_if_triggered")
def test_downgraded_cells_trigger_flag(
    mock_verifier, mock_export, mock_log, mock_threshold, mock_poisson, mock_div_stats, mock_validate,
    base_config, ensure_output_dir_exists
):
    downgraded = make_cell(0.0, 0.0, 0.0, velocity=None, fluid_mask=False)
    updated = make_cell(0.0, 0.0, 0.0, pressure=1.0)

    mock_div_stats.return_value = {"divergence": [], "max": 0.0}
    mock_poisson.return_value = ([updated], False, {})
    mock_threshold.return_value = 0.1

    apply_pressure_correction([downgraded], base_config, step=3, output_folder=str(ensure_output_dir_exists))

    positional_args = mock_verifier.call_args[0]
    assert len(positional_args) == 5
    triggered_flags = positional_args[4]

    assert isinstance(triggered_flags, list)
    assert "downgraded_cells" in triggered_flags
    assert "no_pressure_mutation" in triggered_flags
    assert "empty_divergence" in triggered_flags



