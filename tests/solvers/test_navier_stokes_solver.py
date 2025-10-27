import pathlib
import pytest
from unittest.mock import patch
from src.solvers.navier_stokes_solver import solve_navier_stokes_step
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True):
    """Helper function to create a cell with default values."""
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )

@pytest.fixture
def base_config():
    """Provides a consistent base configuration dictionary."""
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
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01
        }
    }

# FIX: Re-introducing the fixture to ensure hardcoded I/O paths (like 'data/snapshots') 
# exist before the tests run. This resolves the FileNotFoundError that occurs 
# when non-mocked internal functions attempt I/O.
@pytest.fixture(autouse=True)
def ensure_output_dir_exists():
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    (root / "data" / "snapshots").mkdir(parents=True, exist_ok=True)
    # Ensure the general data path exists as well
    (root / "data").mkdir(parents=True, exist_ok=True)


# FIX: Mocking compute_divergence_stats is necessary to suppress I/O inside apply_pressure_correction
@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.diagnostics.navier_stokes_verifier.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
def test_solver_pipeline_executes_all_steps(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config, tmp_path):
    """
    Tests that the main solver function executes the three core steps 
    (momentum, pressure, projection) and packages metadata correctly.
    """
    # Create the temporary output path and pass it to the verifier mock
    temp_output_dir = tmp_path / "navier_stokes_output"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # Mock return for mock_div_stats must be consistent with the 2 fluid cells in the grid.
    mock_div_stats.return_value = {"divergence": [0.01, 0.02], "max": 0.02}

    initial_grid = [
        make_cell(0.0, 0.0, 0.0),
        make_cell(1.0, 0.0, 0.0)
    ]

    grid_after_momentum = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.0, 0.0])
    ]
    grid_after_pressure = [
        make_cell(0.0, 0.0, 0.0, velocity=[0.9, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 0.9, 0.0])
    ]
    grid_after_projection = [
        make_cell(0.0, 0.0, 0.0, velocity=[0.8, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 0.8, 0.0])
    ]

    mock_momentum.return_value = grid_after_momentum
    # Mocked data shows pressure mutation occurred (count=1, True)
    mock_pressure.return_value = (grid_after_pressure, True, 2, {"pressure_mutation_count": 1, "divergence": [0.01, 0.02]})
    mock_projection.return_value = grid_after_projection

    # Pass the temporary path to the solver, which forwards it to the verifier
    result_grid, metadata = solve_navier_stokes_step(initial_grid, base_config, step_index=5, output_folder=temp_output_dir)

    assert result_grid[0].velocity == [0.8, 0.0, 0.0]
    assert result_grid[1].velocity == [0.0, 0.8, 0.0]
    assert metadata["pressure_mutated"] is True
    assert metadata["projection_passes"] == 2
    assert metadata["pressure_mutation_count"] == 1
    assert metadata["divergence"] == [0.01, 0.02]

    # Ensure the verifier was called with the correct temporary path
    mock_verifier.assert_called_once()
    assert mock_verifier.call_args[1]["output_folder"] == str(temp_output_dir)


@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.diagnostics.navier_stokes_verifier.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
def test_triggered_flags_are_detected(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config, tmp_path):
    """
    Tests that the solver correctly identifies conditions that trigger verification flags, 
    such as zero mutations, zero divergence entries, or downgraded cells.
    """
    # Create the temporary output path and pass it to the verifier mock
    temp_output_dir = tmp_path / "navier_stokes_output"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # FIX: Set mock_div_stats return value to be CONSISTENT with the 1 fluid cell [0.0].
    # This bypasses the real validation crash.
    mock_div_stats.return_value = {"divergence": [0.0], "max": 0.0}

    # Cell 1: velocity is [0,0,0], fluid_mask=True (1 fluid cell)
    # Cell 2: fluid_mask=False (Triggers 'downgraded_cells' flag)
    initial_grid = [
        make_cell(0.0, 0.0, 0.0, velocity=None, fluid_mask=True),
        make_cell(1.0, 0.0, 0.0, velocity=None, fluid_mask=False)
    ]

    mock_momentum.return_value = initial_grid
    # The pressure mock *still* returns an empty list in its metadata, 
    # which is what triggers the 'empty_divergence' flag in navier_stokes_solver.
    mock_pressure.return_value = (initial_grid, False, 0, {"pressure_mutation_count": 0, "divergence": []})
    mock_projection.return_value = initial_grid

    # Pass the temporary path to the solver, which forwards it to the verifier
    solve_navier_stokes_step(initial_grid, base_config, step_index=6, output_folder=temp_output_dir)

    args = mock_verifier.call_args[1]
    assert isinstance(args, dict)
    assert args.get("triggered_flags") is not None

    # Check for all three expected flags
    assert "no_pressure_mutation" in args["triggered_flags"]
    assert "empty_divergence" in args["triggered_flags"]
    assert "downgraded_cells" in args["triggered_flags"]
    assert mock_verifier.call_args[1]["output_folder"] == str(temp_output_dir)


@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.diagnostics.navier_stokes_verifier.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats")
def test_solver_returns_grid_and_metadata(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config, tmp_path):
    """Tests the structure and content of the final return values (grid and metadata)."""
    # Create the temporary output path and pass it to the verifier mock
    temp_output_dir = tmp_path / "navier_stokes_output"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # Mock return for the newly patched I/O function (consistent with 1 fluid cell)
    mock_div_stats.return_value = {"divergence": [0.01], "max": 0.01}

    grid = [make_cell(0.0, 0.0, 0.0)]
    mock_momentum.return_value = grid
    # Mocked data shows pressure mutation occurred (count=2, True)
    mock_pressure.return_value = (grid, True, 1, {"pressure_mutation_count": 2, "divergence": [0.01]})
    mock_projection.return_value = grid

    # Pass the temporary path to the solver, which forwards it to the verifier
    result_grid, metadata = solve_navier_stokes_step(grid, base_config, step_index=7, output_folder=temp_output_dir)
    assert isinstance(result_grid, list)
    assert isinstance(metadata, dict)
    assert metadata["pressure_mutation_count"] == 2
    assert metadata["divergence"] == [0.01]
    assert "no_pressure_mutation" not in metadata 
    assert "no_pressure_mutation" not in mock_verifier.call_args[1]["triggered_flags"]

    # Check that the grid returned is the final projection grid
    mock_projection.assert_called_once_with(grid, base_config)
    assert result_grid is grid
    assert mock_verifier.call_args[1]["output_folder"] == str(temp_output_dir)


