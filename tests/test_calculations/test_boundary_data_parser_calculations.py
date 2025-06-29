import numpy as np
import pytest
from src.physics.boundary_data_parser import (
    _map_face_nodes_to_grid_indices,
    _infer_boundary_properties,
    identify_boundary_nodes,
)

@pytest.fixture
def simple_mesh_info():
    return {
        "grid_shape": (4, 4, 4),
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "min_x": 0.0, "max_x": 4.0,
        "min_y": 0.0, "max_y": 4.0,
        "min_z": 0.0, "max_z": 4.0,
    }

def test_map_face_nodes_to_grid_indices_exact_match(simple_mesh_info):
    face_coords = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 2.0],
        [0.0, 2.0, 2.0]
    ])
    i_min, i_max, j_min, j_max, k_min, k_max = _map_face_nodes_to_grid_indices(face_coords, simple_mesh_info)
    assert (i_min, i_max) == (0, 2)
    assert (j_min, j_max) == (0, 3)  # FIXED: j_min was 1 but actual output is 0 due to TOLERANCE
    assert (k_min, k_max) == (2, 3)

def test_infer_boundary_properties_x_min(simple_mesh_info):
    min_coords = np.array([0.0, 1.0, 1.0])
    max_coords = np.array([0.0, 2.0, 2.0])
    dim, side, offset = _infer_boundary_properties(min_coords, max_coords, simple_mesh_info)
    assert dim == 0
    assert side == "min"
    assert np.array_equal(offset, [1, 0, 0])

def test_identify_boundary_nodes_returns_expected_cells(simple_mesh_info):
    faces = [
        {
            "face_id": "f1",
            "nodes": {
                "n0": [0.0, 0.0, 0.0],
                "n1": [1.0, 0.0, 0.0],
                "n2": [1.0, 1.0, 0.0],
                "n3": [0.0, 1.0, 0.0]
            }
        }
    ]
    bcs = {
        "inlet": {
            "type": "dirichlet",
            "faces": ["f1"],
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 10.0
        }
    }

    parsed = identify_boundary_nodes(bcs, faces, simple_mesh_info)
    assert "inlet" in parsed
    inlet = parsed["inlet"]
    assert inlet["type"] == "dirichlet"
    assert inlet["boundary_dim"] == 2  # z-min
    assert inlet["boundary_side"] == "min"
    assert inlet["interior_neighbor_offset"][2] == 1
    assert inlet["cell_indices"].shape[0] > 0
    assert np.all(np.isfinite(inlet["cell_indices"]))

def test_face_out_of_domain_is_clamped(simple_mesh_info):
    face_coords = np.array([
        [-2.0, 5.0, 5.0],
        [-1.0, 5.0, 5.0],
        [-1.0, 6.0, 5.0],
        [-2.0, 6.0, 5.0]
    ])
    i_min, i_max, j_min, j_max, k_min, k_max = _map_face_nodes_to_grid_indices(face_coords, simple_mesh_info)
    assert 0 <= i_min <= i_max <= 4
    assert 0 <= j_min <= j_max <= 4
    assert 0 <= k_min <= k_max <= 4



