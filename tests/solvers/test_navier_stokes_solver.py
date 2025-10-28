import pathlib
import pytest
from unittest.mock import patch
from src.solvers.navier_stokes_solver import solve_navier_stokes_step
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity if velocity is not None else [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )

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
            "output_interval": 1,
            "disable_io_for_testing": True
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01
        }
    }

@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.solvers.pressure_solver.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
def test_solver_pipeline_executes_all_steps(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config, tmp_path):
    temp_output_dir = tmp_path / "navier_stokes_output"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    mock_div_stats.return_value = {"divergence": [0.01, 0.02], "max": 0.02}

    initial_grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[10.0, 10.0, 10.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[10.0, 10.0, 10.0])
    ]

    grid_after_momentum = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=0.1),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.0, 0.0], pressure=0.2)
    ]
    grid_after_pressure = [
        make_cell(0.0, 0.0, 0.0, velocity=[0.9, 0.0, 0.0], pressure=-0.5),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 0.9, 0.0], pressure=-0.6)
    ]
    grid_after_projection = [
        make_cell(0.0, 0.0, 0.0, velocity=[0.8, 0.0, 0.0], pressure=-0.5),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 0.8, 0.0], pressure=-0.6)
    ]

    mock_momentum.return_value = grid_after_momentum
    mock_pressure.return_value = (grid_after_pressure, True, 2, {"pressure_mutation_count": 1, "divergence": [0.01, 0.02]})
    mock_projection.return_value = grid_after_projection

    result_grid, metadata = solve_navier_stokes_step(initial_grid, base_config, step_index=5, output_folder=temp_output_dir)

    assert result_grid[0].velocity == [0.8, 0.0, 0.0]
    assert result_grid[1].velocity == [0.0, 0.8, 0.0]
    assert result_grid is grid_after_projection
    assert metadata["pressure_mutated"] is True
    assert metadata["projection_passes"] == 2
    assert metadata["pressure_mutation_count"] == 1
    assert metadata["divergence"] == [0.01, 0.02]
    mock_verifier.assert_called_once()
    assert mock_verifier.call_args[0][3] == "data/testing-input-output/navier_stokes_output"

@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.solvers.pressure_solver.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
def test_triggered_flags_are_detected(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config, tmp_path):
    temp_output_dir = tmp_path / "navier_stokes_output"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    mock_div_stats.return_value = {"divergence": [0.0, 0.0], "max": 0.0}

    initial_grid = [
        make_cell(0.0, 0.0, 0.0, velocity=None, fluid_mask=True),
        make_cell(1.0, 0.0, 0.0, velocity=None, fluid_mask=False),
        make_cell(2.0, 0.0, 0.0, velocity=None, fluid_mask=True)
    ]

    mock_momentum.return_value = initial_grid
    mock_pressure.return_value = (initial_grid, False, 0, {"pressure_mutation_count": 0, "divergence": [0.0, 0.0]})
    mock_projection.return_value = initial_grid

    solve_navier_stokes_step(initial_grid, base_config, step_index=6, output_folder=temp_output_dir)

    mock_verifier.assert_called_once()
    triggered_flags = mock_verifier.call_args[0][4]
    assert mock_verifier.call_args[0][3] == "data/testing-input-output/navier_stokes_output"
    assert "no_pressure_mutation" in triggered_flags
    assert "empty_divergence" not in triggered_flags
    assert "downgraded_cells" in triggered_flags

@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.solvers.pressure_solver.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
def test_solver_returns_grid_and_metadata(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config, tmp_path):
    temp_output_dir = tmp_path / "navier_stokes_output"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    mock_div_stats.return_value = {"divergence": [0.01], "max": 0.01}

    grid = [make_cell(0.0, 0.0, 0.0)]
    mock_momentum.return_value = grid
    mock_meta = {"pressure_mutation_count": 1, "divergence": [0.01]}
    mock_pressure.return_value = (grid, True, 1, mock_meta)
    mock_projection.return_value = grid

    result_grid, metadata = solve_navier_stokes_step(grid, base_config, step_index=7, output_folder=temp_output_dir)

    assert isinstance(result_grid, list)
    assert isinstance(metadata, dict)
    assert metadata["pressure_mutation_count"] == 1
    assert metadata["projection_passes"] == 1
    assert metadata.get("divergence", []) == [0.01]
    assert "no_pressure_mutation" not in metadata
    mock_projection.assert_called_once_with(grid, base_config)
    assert result_grid is grid
    assert mock_verifier.call_args[0][3] == "data/testing-input-output/navier_stokes_output"



