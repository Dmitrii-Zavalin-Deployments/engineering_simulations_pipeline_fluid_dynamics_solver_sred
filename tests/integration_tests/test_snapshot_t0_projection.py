# tests/test_snapshot_t0_projection.py
# 🧪 Projection Validation: pressure/velocity effects at t=0

import math
from tests.snapshot_t0_shared import (
    snapshot,
    config,
    domain,
    expected_mask,
    expected_pressure,
    expected_velocity,  # ✅ Added fixture linkage
    tolerances,
    get_domain_cells,
    is_close
)

def test_pressure_field_changes_if_projected(snapshot, domain, expected_mask, expected_pressure, tolerances):
    if snapshot.get("projection_passes", 0) == 0:
        return  # Skip if no projection applied

    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid:
            actual = cell["pressure"]
            # Assert pressure moved at least one tolerance away from initial
            assert not is_close(actual, expected_pressure, tolerances["pressure"]), f"❌ Pressure did not change: {actual}"

def test_velocity_field_changes_if_projected(snapshot, domain, expected_mask, expected_velocity, tolerances):
    if snapshot.get("projection_passes", 0) == 0:
        return  # Skip if projection not performed

    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid:
            actual_v = cell["velocity"]
            assert isinstance(actual_v, list) and len(actual_v) == 3
            # ✅ Updated drift-aware logic
            drift_detected = any(
                abs(actual - expected) > 1e-5
                for actual, expected in zip(actual_v, expected_velocity)
            )
            if not drift_detected:
                print(f"⚠️ Velocity drift too small to register: {actual_v} vs {expected_velocity}")
            assert drift_detected, f"❌ Velocity unchanged: {actual_v}"



