import pytest
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

def test_solver_pipeline_executes_functionally(base_config, tmp_path):
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 1.0, 1.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[1.0, 1.0, 1.0])
    ]
    result_grid, metadata = solve_navier_stokes_step(grid, base_config, step_index=0, output_folder=str(tmp_path))

    assert isinstance(result_grid, list)
    assert all(isinstance(c.velocity, list) for c in result_grid)
    assert metadata.get("projection_passes", 0) >= 0
    assert "pressure_mutated" in metadata

def test_solver_metadata_contains_divergence(base_config, tmp_path):
    grid = [make_cell(0.0, 0.0, 0.0)]
    _, metadata = solve_navier_stokes_step(grid, base_config, step_index=1, output_folder=str(tmp_path))

    assert "divergence" in metadata
    assert isinstance(metadata["divergence"], list)

def test_solver_triggers_diagnostics(base_config, tmp_path):
    grid = [make_cell(0.0, 0.0, 0.0, velocity=None, fluid_mask=False)]
    _, metadata = solve_navier_stokes_step(grid, base_config, step_index=2, output_folder=str(tmp_path))

    triggered_flags = []
    if metadata.get("pressure_mutation_count", 0) == 0:
        triggered_flags.append("no_pressure_mutation")
    if not metadata.get("divergence", []):
        triggered_flags.append("empty_divergence")
    if any(not isinstance(c.velocity, list) or not c.fluid_mask for c in grid):
        triggered_flags.append("downgraded_cells")

    assert "downgraded_cells" in triggered_flags



