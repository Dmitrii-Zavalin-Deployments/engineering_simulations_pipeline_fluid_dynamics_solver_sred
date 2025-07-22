# tests/test_snapshot_structure.py
# 🧪 Modular snapshot structure validation (schema, coordinates, diagnostics)

import math
from tests.snapshot_t0_shared import (
    snapshot,
    config,
    domain,
    get_domain_cells
)

EPSILON = 1e-6

def test_cell_coordinates_match_domain(snapshot, domain):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    x_centers = [domain["min_x"] + (i + 0.5) * dx for i in range(domain["nx"])]
    y_centers = [domain["min_y"] + (j + 0.5) * dy for j in range(domain["ny"])]
    z_centers = [domain["min_z"] + (k + 0.5) * dz for k in range(domain["nz"])]

    expected_coords = [(x, y, z) for x in x_centers for y in y_centers for z in z_centers]
    actual_coords = [(c["x"], c["y"], c["z"]) for c in get_domain_cells(snapshot, domain)]

    for target in expected_coords:
        found = any(
            math.isclose(target[0], a[0], abs_tol=EPSILON) and
            math.isclose(target[1], a[1], abs_tol=EPSILON) and
            math.isclose(target[2], a[2], abs_tol=EPSILON)
            for a in actual_coords
        )
        assert found, f"❌ Missing expected cell center: {target}"

def test_field_types_are_consistent(snapshot, domain):
    domain_cells = get_domain_cells(snapshot, domain)
    for cell in domain_cells:
        assert isinstance(cell["x"], (int, float)), "❌ Invalid x coordinate type"
        assert isinstance(cell["y"], (int, float)), "❌ Invalid y coordinate type"
        assert isinstance(cell["z"], (int, float)), "❌ Invalid z coordinate type"
        assert isinstance(cell["fluid_mask"], bool), "❌ fluid_mask must be boolean"
        if cell["velocity"] is not None:
            assert isinstance(cell["velocity"], list), "❌ velocity must be list or None"
            assert len(cell["velocity"]) == 3
            assert all(isinstance(v, (int, float)) for v in cell["velocity"]), "❌ velocity components must be numeric"
        if cell["pressure"] is not None:
            assert isinstance(cell["pressure"], (int, float)), "❌ pressure must be numeric or None"

def test_snapshot_diagnostics_present(snapshot):
    assert isinstance(snapshot.get("reflex_score"), (float, int)), "❌ reflex_score must be numeric"
    assert isinstance(snapshot.get("projection_passes"), int), "❌ projection_passes must be integer"
    assert snapshot.get("projection_passes", 0) >= 0, "❌ projection_passes must be non-negative"
    assert "divergence_zero" in snapshot, "❌ Missing divergence_zero flag"
    assert isinstance(snapshot.get("divergence_zero"), bool), "❌ divergence_zero must be boolean"



