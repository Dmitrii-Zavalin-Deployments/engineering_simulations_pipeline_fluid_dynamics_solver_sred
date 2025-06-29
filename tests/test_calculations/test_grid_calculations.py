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
    assert np.all(info["cell_centers"] > 0)

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

    # MAC grid shapes with ghost cells (gx = 1)
    # u: (nx + 1 + 2*gx, ny + 2*gx, nz + 2*gx)
    # v: (nx + 2*gx, ny + 1 + 2*gx, nz + 2*gx)
    # w: (nx + 2*gx, ny + 2*gx, nz + 1 + 2*gx)
    # p: (nx + 2*gx, ny + 2*gx, nz + 2*gx)

    assert fields["u"].shape == (6, 5, 5)
    assert fields["v"].shape == (5, 6, 5)
    assert fields["w"].shape == (5, 5, 6)
    assert fields["p"].shape == (5, 5, 5)

    assert all(f.dtype == np.float64 for f in fields.values())
    assert all(np.allclose(f, 0.0) for f in fields.values())

def test_generate_physical_coordinates_yields_correct_offsets():
    coords = generate_physical_coordinates((2, 2, 2), (1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), ghost_width=1)

    ux, uy, uz = coords["u"]  # shape: (5,)
    vx, vy, vz = coords["v"]
    wx, wy, wz = coords["w"]
    px, py, pz = coords["p"]

    # staggered field lengths (nx=2, gx=1)
    assert ux.shape == (5,)
    assert uy.shape == (5,)
    assert uz.shape == (5,)
    assert vx.shape == (5,)
    assert vy.shape == (5,)
    assert vz.shape == (5,)
    assert wx.shape == (5,)
    assert wy.shape == (5,)
    assert wz.shape == (5,)
    assert px.shape == (5,)
    assert py.shape == (5,)
    assert pz.shape == (5,)

    # check expected offset values for start of axes
    assert pytest.approx(ux[0]) == -1.0
    assert pytest.approx(vy[0]) == -1.0
    assert pytest.approx(wz[0]) == -1.0
    assert pytest.approx(px[0]) == -0.5



