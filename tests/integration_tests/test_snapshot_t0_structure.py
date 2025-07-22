# tests/test_snapshot_t0_structure.py
# 🧪 Structure and Grid Validation for t=0 Snapshot

import math
from tests.integration_tests.snapshot_t0_shared import (
    snapshot,
    config,
    domain,
    expected_mask,
    get_domain_cells
)

EXPECTED_STEP_INDEX = 0
EPSILON = 1e-6

def test_step_index(snapshot):
    assert snapshot["step_index"] == EXPECTED_STEP_INDEX

def test_grid_structure(snapshot):
    grid = snapshot["grid"]
    assert isinstance(grid, list), "❌ Grid should be a list"
    for cell in grid:
        assert isinstance(cell, dict), "❌ Each grid cell should be a dictionary"
        for key in ["x", "y", "z", "velocity", "pressure", "fluid_mask"]:
            assert key in cell, f"❌ Missing key '{key}' in cell"

        velocity = cell.get("velocity")
        if snapshot.get("damping_enabled") and isinstance(velocity, list) and len(velocity) == 3:
            if max(abs(v) for v in velocity) < 1e-4 and cell.get("fluid_mask"):
                print(f"🔕 Suppressed velocity at ({cell['x']}, {cell['y']}, {cell['z']}) → {velocity}")

def test_grid_size_matches_mask(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    assert len(domain_cells) == len(expected_mask), "❌ Domain cell count mismatch vs. input mask"

def test_cell_coordinates(snapshot, domain):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    x_centers = [domain["min_x"] + (i + 0.5) * dx for i in range(domain["nx"])]
    y_centers = [domain["min_y"] + (j + 0.5) * dy for j in range(domain["ny"])]
    z_centers = [domain["min_z"] + (k + 0.5) * dz for k in range(domain["nz"])]

    expected_coords = [(x, y, z) for x in x_centers for y in y_centers for z in z_centers]
    actual_coords = [(c["x"], c["y"], c["z"]) for c in get_domain_cells(snapshot, domain)]

    for expected in expected_coords:
        matches = [
            math.isclose(expected[0], a[0], abs_tol=EPSILON) and
            math.isclose(expected[1], a[1], abs_tol=EPSILON) and
            math.isclose(expected[2], a[2], abs_tol=EPSILON)
            for a in actual_coords
        ]
        assert any(matches), f"❌ Missing expected coordinate: {expected}"

def test_fluid_mask_matches_geometry(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    actual_mask = [c["fluid_mask"] for c in domain_cells]
    assert actual_mask == expected_mask, "❌ Fluid mask mismatch against input geometry"



