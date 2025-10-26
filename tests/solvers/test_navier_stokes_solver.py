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

@pytest.fixture(autouse=True)
def ensure_output_dir_exists():
    """
    Ensures the necessary output directories exist, although mocking I/O is preferred 
    for robust unit testing, this is kept for system-level consistency.
    """
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    (root / "data" / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / "data" / "testing-input-output" / "navier_stokes_output").mkdir(parents=True, exist_ok=True)


# FIX: Mocking compute_divergence_stats to prevent FileNotFoundError during test execution,
# as the patch on apply_pressure_correction was not fully suppressing the real function's I/O calls.
@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.diagnostics.navier_stokes_verifier.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats") # FIX: New patch for I/O suppression
def test_solver_pipeline_executes_all_steps(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config):
    """
    Tests that the main solver function executes the three core steps 
    (momentum, pressure, projection) and packages metadata correctly.
    """
    # Mock return for the newly patched I/O function
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

    result_grid, metadata = solve_navier_stokes_step(initial_grid, base_config, step_index=5)

    assert result_grid[0].velocity == [0.8, 0.0, 0.0]
    assert result_grid[1].velocity == [0.0, 0.8, 0.0]
    assert metadata["pressure_mutated"] is True
    assert metadata["projection_passes"] == 2
    assert metadata["pressure_mutation_count"] == 1
    assert metadata["divergence"] == [0.01, 0.02]
    mock_verifier.assert_called_once()


@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.diagnostics.navier_stokes_verifier.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats") # FIX: New patch for I/O suppression
def test_triggered_flags_are_detected(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config):
    """
    Tests that the solver correctly identifies conditions that trigger verification flags, 
    such as zero mutations, zero divergence entries, or downgraded cells.
    """
    # Mock return for the newly patched I/O function
    mock_div_stats.return_value = {"divergence": [], "max": 0.0}

    # Cell 1: velocity is [0,0,0], fluid_mask=True (OK)
    # Cell 2: fluid_mask=False (Triggers 'downgraded_cells' flag via 'not c.fluid_mask')
    initial_grid = [
        make_cell(0.0, 0.0, 0.0, velocity=None, fluid_mask=True),
        make_cell(1.0, 0.0, 0.0, velocity=None, fluid_mask=False)
    ]

    mock_momentum.return_value = initial_grid
    # Mocked data results in:
    # 1. pressure_mutation_count=0 -> Triggers 'no_pressure_mutation'
    # 2. divergence=[] -> Triggers 'empty_divergence'
    mock_pressure.return_value = (initial_grid, False, 0, {"pressure_mutation_count": 0, "divergence": []})
    mock_projection.return_value = initial_grid

    solve_navier_stokes_step(initial_grid, base_config, step_index=6)

    args = mock_verifier.call_args[1]
    assert isinstance(args, dict)
    assert args.get("triggered_flags") is not None
    
    # Check for all three expected flags
    assert "no_pressure_mutation" in args["triggered_flags"]
    assert "empty_divergence" in args["triggered_flags"]
    assert "downgraded_cells" in args["triggered_flags"]


@patch("src.solvers.momentum_solver.apply_momentum_update")
@patch("src.solvers.pressure_solver.apply_pressure_correction")
@patch("src.physics.velocity_projection.apply_pressure_velocity_projection")
@patch("src.diagnostics.navier_stokes_verifier.run_verification_if_triggered")
@patch("src.solvers.pressure_solver.compute_divergence_stats") # FIX: New patch for I/O suppression
def test_solver_returns_grid_and_metadata(mock_div_stats, mock_verifier, mock_projection, mock_pressure, mock_momentum, base_config):
    """Tests the structure and content of the final return values (grid and metadata)."""
    # Mock return for the newly patched I/O function
    mock_div_stats.return_value = {"divergence": [0.01], "max": 0.01}

    grid = [make_cell(0.0, 0.0, 0.0)]
    mock_momentum.return_value = grid
    # Mocked data shows pressure mutation occurred (count=2, True)
    mock_pressure.return_value = (grid, True, 1, {"pressure_mutation_count": 2, "divergence": [0.01]})
    mock_projection.return_value = grid

    result_grid, metadata = solve_navier_stokes_step(grid, base_config, step_index=7)
    assert isinstance(result_grid, list)
    assert isinstance(metadata, dict)
    assert metadata["pressure_mutation_count"] == 2
    assert metadata["divergence"] == [0.01]
    assert "no_pressure_mutation" not in metadata # Should not be in metadata, but should not be a triggered flag either
    assert "no_pressure_mutation" not in mock_verifier.call_args[1]["triggered_flags"]
    
    # Check that the grid returned is the final projection grid
    mock_projection.assert_called_once_with(grid, base_config)
    assert result_grid is grid



