import numpy as np
import pytest
from src.utils.grid import (
    create_structured_grid_info,
    get_cell_centers,
    create_mac_grid_fields,
    generate_physical_coordinates
)

def test_create_structured_grid_info_outputs_valid_geometry():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 2.0,
        "min_z": 0.0, "max_z": 3.0,
        "nx": 2, "ny": 4, "nz": 6
    }

    info = create_structured_grid_info(domain)

    assert info["dx"] == pytest.approx(0.5)
    assert info["dy"] == pytest.approx(0.5)
    assert info["dz"] == pytest.approx(0.5)
    assert info["num_cells"] == 48
    assert info["cell_centers"].shape == (48, 3)
    assert np.all(info["cell_centers"] > 0)  # non-zero coordinates

def test_get_cell_centers_computes_centers_correctly():
    centers = get_cell_centers(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2)
    expected = np.array([
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.75],
        [0.25, 0.75, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.25],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
        [0.75, 0.75, 0.75],
    ])
    assert centers.shape == (8, 3)
    assert np.allclose(centers, expected)

def test_create_mac_grid_fields_returns_correct_shapes():
    fields = create_mac_grid_fields((3, 3, 3), ghost_width=1)

    assert fields["u"].shape == (5, 5, 5)
    assert fields["v"].shape == (5, 5, 5)
    assert fields["w"].shape == (5, 5, 5)
    assert fields["p"].shape == (5, 5, 5)

    assert all(v.dtype == np.float64 for v in fields.values())
    assert all(np.allclose(v, 0.0) for v in fields.values())

def test_generate_physical_coordinates_yields_correct_offsets():
    coords = generate_physical_coordinates((2, 2, 2), (1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), ghost_width=1)

    ux, uy, uz = coords["u"]
    vx, vy, vz = coords["v"]
    wx, wy, wz = coords["w"]
    px, py, pz = coords["p"]

    assert ux.shape == (6,)
    assert uy.shape == (6,)
    assert uz.shape == (6,)
    assert pytest.approx(ux[0]) == -1.0
    assert pytest.approx(vy[0]) == -1.0
    assert pytest.approx(wz[0]) == -1.0
    assert pytest.approx(px[0]) == -0.5



