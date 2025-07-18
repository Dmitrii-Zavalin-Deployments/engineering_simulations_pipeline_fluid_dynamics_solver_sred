# tests/test_snapshot_t0_projection.py
# 🧪 Projection Behavior Validation for t=0 Snapshot

import math
from tests.snapshot_t0_shared import (
    snapshot,
    domain,
    expected_mask,
    expected_velocity,
    expected_pressure,
    tolerances,
    get_domain_cells
)

def test_pressure_projection_mutated(snapshot):
    mutated_flag = snapshot.get("pressure_mutated", None)
    assert mutated_flag in [True, False], "❌ pressure_mutated flag should be boolean"

def test_velocity_projection_applied(snapshot):
    assert snapshot.get("velocity_projected", True) is True, "❌ velocity_projected should be True if projection occurred"

def test_pressure_field_changes_if_projected(snapshot, domain, expected_mask, expected_pressure, tolerances):
    domain_cells = get_domain_cells(snapshot, domain)
    if snapshot.get("projection_passes", 0) > 0 and not snapshot.get("projection_skipped", False):
        fluid_pressures = [
            cell["pressure"] for cell, is_fluid in zip(domain_cells, expected_mask) if is_fluid
        ]
        deltas = [abs(p - expected_pressure) for p in fluid_pressures]
        assert any(d > tolerances["pressure"] for d in deltas), "❌ Pressure field unchanged after projection"

def test_velocity_field_changes_if_projected(snapshot, domain, expected_mask, expected_velocity, tolerances):
    domain_cells = get_domain_cells(snapshot, domain)
    if snapshot.get("projection_passes", 0) > 0 and snapshot.get("velocity_projected", True):
        fluid_velocities = [
            cell["velocity"] for cell, is_fluid in zip(domain_cells, expected_mask) if is_fluid
        ]
        for v in fluid_velocities:
            assert isinstance(v, list), "❌ Velocity must be a list"
        deltas = [
            math.sqrt(sum((a - b) ** 2 for a, b in zip(v, expected_velocity)))
            for v in fluid_velocities
        ]
        assert any(d > tolerances["velocity"] for d in deltas), "❌ Velocity field unchanged after projection"



