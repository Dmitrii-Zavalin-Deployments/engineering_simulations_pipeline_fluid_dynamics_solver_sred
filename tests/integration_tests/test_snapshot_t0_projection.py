# tests/test_snapshot_t0_projection.py
# 🧪 Projection Validation: pressure/velocity effects at t=0

import math
from tests.integration_tests.snapshot_t0_shared import (
    snapshot,
    config,
    domain,
    expected_mask,
    expected_pressure,
    expected_velocity,
    tolerances,
    get_domain_cells,
    is_close
)

def vector_magnitude(vec):
    if isinstance(vec, list) and len(vec) == 3:
        return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    raise ValueError(f"Invalid vector: {vec}")

def test_pressure_field_changes_if_projected(snapshot, domain, expected_mask, expected_pressure, tolerances):
    if snapshot.get("projection_passes", 0) == 0:
        return  # Skip if no projection applied

    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid:
            actual = cell["pressure"]
            assert not is_close(actual, expected_pressure, tolerances["pressure"]), f"❌ Pressure did not change: {actual}"

def test_velocity_field_changes_if_projected(snapshot, domain, expected_mask, expected_velocity, tolerances):
    if snapshot.get("projection_passes", 0) == 0:
        return  # Skip if projection not performed

    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid:
            velocity = cell.get("velocity")
            assert isinstance(velocity, list) and len(velocity) == 3
            assert all(isinstance(v, (int, float)) for v in velocity)

            drift_vector = [abs(a - b) for a, b in zip(velocity, expected_velocity)]
            drift_magnitude = vector_magnitude(drift_vector)

            if snapshot.get("damping_enabled") and vector_magnitude(velocity) < 1e-4:
                print(f"🔕 Suppressed velocity in projected cell at ({cell['x']}, {cell['y']}, {cell['z']}): {velocity}")

            if drift_magnitude < 1e-5:
                print(f"⚠️ Drift too small to register at ({cell['x']}, {cell['y']}, {cell['z']}): {velocity} vs {expected_velocity}")

            assert drift_magnitude > 1e-5, f"❌ Velocity unchanged after projection: {velocity}"



