# tests/metrics/test_divergence_metrics.py
# 🧪 Unit tests for src/metrics/divergence_metrics.py

from src.grid_modules.cell import Cell
from src.metrics.divergence_metrics import compute_max_divergence

def make_cell(x, y, z, vx=0.0, vy=0.0, vz=0.0, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=[vx, vy, vz], pressure=0.0, fluid_mask=fluid)

def test_divergence_returns_zero_for_empty_grid():
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 1,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    assert compute_max_divergence([], domain) == 0.0

def test_divergence_returns_zero_if_no_fluid_cells():
    grid = [make_cell(0,0,0, fluid=False), make_cell(1,0,0, fluid=False)]
    domain = {"min_x": 0.0, "max_x": 2.0, "nx": 2,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    assert compute_max_divergence(grid, domain) == 0.0

def test_divergence_returns_zero_without_valid_neighbors():
    grid = [make_cell(0.0, 0.0, 0.0)]
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 1,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    assert compute_max_divergence(grid, domain) == 0.0

def test_divergence_computes_expected_value_for_symmetric_neighbors():
    dx = 1.0
    cell_center = make_cell(1.0, 1.0, 1.0)
    cell_x_minus = make_cell(0.0, 1.0, 1.0, vx=1.0)
    cell_x_plus  = make_cell(2.0, 1.0, 1.0, vx=3.0)
    cell_y_minus = make_cell(1.0, 0.0, 1.0, vy=2.0)
    cell_y_plus  = make_cell(1.0, 2.0, 1.0, vy=4.0)
    cell_z_minus = make_cell(1.0, 1.0, 0.0, vz=6.0)
    cell_z_plus  = make_cell(1.0, 1.0, 2.0, vz=10.0)

    grid = [cell_center, cell_x_minus, cell_x_plus,
            cell_y_minus, cell_y_plus, cell_z_minus, cell_z_plus]

    domain = {"min_x": 0.0, "max_x": 3.0, "nx": 3,
              "min_y": 0.0, "max_y": 3.0, "ny": 3,
              "min_z": 0.0, "max_z": 3.0, "nz": 3}

    # dx = dy = dz = 1.0
    # divergence = [(3-1)/2 + (4-2)/2 + (10-6)/2] = (2 + 2 + 4)/2 = 4.0
    assert compute_max_divergence(grid, domain) == 4.0

def test_divergence_skips_nonfluid_neighbors():
    center = make_cell(1.0, 1.0, 1.0)
    fluid_x_minus = make_cell(0.0, 1.0, 1.0, vx=1.0)
    solid_x_plus  = make_cell(2.0, 1.0, 1.0, vx=99.0, fluid=False)
    grid = [center, fluid_x_minus, solid_x_plus]

    domain = {"min_x": 0.0, "max_x": 2.0, "nx": 2,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    assert compute_max_divergence(grid, domain) == 0.0

def test_divergence_handles_nonvector_velocity_fields():
    center = make_cell(1.0, 1.0, 1.0)
    center.velocity = None
    left  = make_cell(0.0, 1.0, 1.0, vx=1.0)
    right = make_cell(2.0, 1.0, 1.0, vx=3.0)
    grid = [center, left, right]

    domain = {"min_x": 0.0, "max_x": 3.0, "nx": 3,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    assert compute_max_divergence(grid, domain) == 1.0



