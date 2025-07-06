import pytest
from src.utils import mesh_utils

def test_get_domain_extents_returns_bounds():
    faces = [
        {"nodes": {
            "a": [0.0, 1.0, 2.0],
            "b": [1.0, 1.0, 2.0],
            "c": [1.0, 2.0, 3.0]
        }},
        {"nodes": {
            "d": [0.5, 0.5, 1.0]
        }}
    ]
    min_x, max_x, min_y, max_y, min_z, max_z = mesh_utils.get_domain_extents(faces)

    assert min_x == 0.0
    assert max_x == 1.0
    assert min_y == 0.5
    assert max_y == 2.0
    assert min_z == 1.0
    assert max_z == 3.0

def test_get_domain_extents_raises_on_empty():
    with pytest.raises(ValueError, match="No nodes found"):
        mesh_utils.get_domain_extents([])

def test_infer_parameters_from_json_override():
    dx, nx = mesh_utils.infer_uniform_grid_parameters(
        min_val=0.0,
        max_val=2.0,
        all_coords_for_axis=[0.0, 1.0, 2.0],
        axis_name="x",
        json_nx=4
    )
    assert nx == 4
    assert dx == pytest.approx(0.5)

def test_infer_parameters_from_coords_only():
    spacing, num_cells = mesh_utils.infer_uniform_grid_parameters(
        min_val=0.0,
        max_val=3.0,
        all_coords_for_axis=[0.0, 1.5, 3.0],
        axis_name="y"
    )
    assert num_cells == 2
    assert spacing == pytest.approx(1.5)

def test_infer_parameters_single_unique_coord_with_extent():
    spacing, num_cells = mesh_utils.infer_uniform_grid_parameters(
        min_val=2.0,
        max_val=3.0,
        all_coords_for_axis=[2.0, 2.0, 2.0],
        axis_name="z"
    )
    assert num_cells == 1
    assert spacing == pytest.approx(1.0)

def test_infer_parameters_with_dense_coords_and_tiny_spacing():
    points = [0.0 + i * 1e-5 for i in range(6)]
    spacing, num_cells = mesh_utils.infer_uniform_grid_parameters(
        min_val=0.0,
        max_val=5e-5,
        all_coords_for_axis=points,
        axis_name="x"
    )
    assert num_cells == 5
    assert spacing == pytest.approx(1e-5)

def test_infer_parameters_zero_extent_returns_one_cell():
    spacing, num_cells = mesh_utils.infer_uniform_grid_parameters(
        min_val=1.0,
        max_val=1.0,
        all_coords_for_axis=[1.0, 1.0],
        axis_name="y"
    )
    assert num_cells == 1
    assert spacing == pytest.approx(1.0)



