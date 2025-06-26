# tests/test_grid_initialization/test_field_location_mapping.py

import numpy as np
import pytest

def generate_field_coordinates(nx, ny, nz, dx=1.0, origin=(0, 0, 0)):
    """
    Computes the physical coordinates for each field center:
    - u: faces normal to x, shape (nx+1, ny, nz)
    - v: faces normal to y, shape (nx, ny+1, nz)
    - w: faces normal to z, shape (nx, ny, nz+1)
    - p: cell centers, shape (nx, ny, nz)
    """
    ox, oy, oz = origin
    u_x = np.linspace(ox, ox + nx * dx, nx + 1)
    u_y = np.linspace(oy + 0.5 * dx, oy + (ny - 0.5) * dx, ny)
    u_z = np.linspace(oz + 0.5 * dx, oz + (nz - 0.5) * dx, nz)

    v_x = np.linspace(ox + 0.5 * dx, ox + (nx - 0.5) * dx, nx)
    v_y = np.linspace(oy, oy + ny * dx, ny + 1)
    v_z = np.linspace(oz + 0.5 * dx, oz + (nz - 0.5) * dx, nz)

    w_x = np.linspace(ox + 0.5 * dx, ox + (nx - 0.5) * dx, nx)
    w_y = np.linspace(oy + 0.5 * dx, oy + (ny - 0.5) * dx, ny)
    w_z = np.linspace(oz, oz + nz * dx, nz + 1)

    p_x = np.linspace(ox + 0.5 * dx, ox + (nx - 0.5) * dx, nx)
    p_y = np.linspace(oy + 0.5 * dx, oy + (ny - 0.5) * dx, ny)
    p_z = np.linspace(oz + 0.5 * dx, oz + (nz - 0.5) * dx, nz)

    return {
        "u": (u_x, u_y, u_z),
        "v": (v_x, v_y, v_z),
        "w": (w_x, w_y, w_z),
        "p": (p_x, p_y, p_z),
    }

def test_field_centers_are_staggered_consistently():
    """Ensure each field is offset as expected according to MAC grid."""
    nx, ny, nz = 4, 3, 2
    dx = 1.0
    coords = generate_field_coordinates(nx, ny, nz, dx)

    # Validate field center locations (relative comparisons)
    u_x, u_y, u_z = coords["u"]
    v_x, v_y, v_z = coords["v"]
    w_x, w_y, w_z = coords["w"]
    p_x, p_y, p_z = coords["p"]

    # Check alignment: pressure center vs other fields
    # u and p should differ by 0.5 dx along x
    assert np.isclose(u_x[0], p_x[0] - 0.5 * dx), "u_x not staggered correctly relative to pressure"
    assert np.isclose(v_y[0], p_y[0] - 0.5 * dx), "v_y not staggered correctly relative to pressure"
    assert np.isclose(w_z[0], p_z[0] - 0.5 * dx), "w_z not staggered correctly relative to pressure"

    # Staggered field should span full domain plus 1 cell
    assert len(u_x) == nx + 1
    assert len(v_y) == ny + 1
    assert len(w_z) == nz + 1

    # Pressure field centered at cell centers
    assert len(p_x) == nx
    assert len(p_y) == ny
    assert len(p_z) == nz



