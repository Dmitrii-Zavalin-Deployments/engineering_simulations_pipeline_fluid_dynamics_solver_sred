import pathlib
import pytest
from unittest.mock import patch
from src.solvers.momentum_solver import apply_momentum_update
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True):
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
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    (root / "data" / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / "data" / "testing-input-output" / "navier_stokes_output").mkdir(parents=True, exist_ok=True)

@patch("src.solvers.momentum_solver.compute_advection")
@patch("src.solvers.momentum_solver.apply_viscous_terms")
def test_velocity_updated_for_fluid_cells(mock_viscosity, mock_advection, base_config):
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.0, 0.0])
    ]

    mock_advection.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.1, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.1, 0.0])
    ]
    mock_viscosity.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.2, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.2, 0.0])
    ]

    result = apply_momentum_update(grid, base_config, step=1)
    assert len(result) == 2
    assert result[0].velocity == [1.2, 0.0, 0.0]
    assert result[1].velocity == [0.0, 1.2, 0.0]
    assert result[0].pressure == 0.0
    assert result[1].pressure == 0.0

@patch("src.solvers.momentum_solver.compute_advection")
@patch("src.solvers.momentum_solver.apply_viscous_terms")
def test_solid_cells_are_excluded(mock_viscosity, mock_advection, base_config):
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.0, 0.0])
    ]

    mock_advection.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.1, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.1, 0.0])
    ]
    mock_viscosity.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.2, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.2, 0.0])
    ]

    result = apply_momentum_update(grid, base_config, step=2)
    assert result[0].fluid_mask is False
    assert result[0].velocity is None
    assert result[0].pressure is None
    assert result[1].velocity == [0.0, 1.2, 0.0]

@patch("src.solvers.momentum_solver.compute_advection")
@patch("src.solvers.momentum_solver.apply_viscous_terms")
def test_velocity_preserved_shape_and_order(mock_viscosity, mock_advection, base_config):
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 2.0, 3.0]),
        make_cell(1.0, 1.0, 1.0, velocity=[4.0, 5.0, 6.0])
    ]

    mock_advection.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.1, 2.1, 3.1]),
        make_cell(1.0, 1.0, 1.0, velocity=[4.1, 5.1, 6.1])
    ]
    mock_viscosity.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.2, 2.2, 3.2]),
        make_cell(1.0, 1.0, 1.0, velocity=[4.2, 5.2, 6.2])
    ]

    result = apply_momentum_update(grid, base_config, step=3)
    assert result[0].velocity == [1.2, 2.2, 3.2]
    assert result[1].velocity == [4.2, 5.2, 6.2]

@patch("src.solvers.momentum_solver.compute_advection")
@patch("src.solvers.momentum_solver.apply_viscous_terms")
def test_debug_output_for_fluid_cells(mock_viscosity, mock_advection, base_config, capsys):
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.0, 0.0])
    ]

    mock_advection.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.1, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.1, 0.0])
    ]
    mock_viscosity.return_value = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.2, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.2, 0.0])
    ]

    apply_momentum_update(grid, base_config, step=4)
    captured = capsys.readouterr()
    assert "[MOMENTUM] Fluid cell @" in captured.out



