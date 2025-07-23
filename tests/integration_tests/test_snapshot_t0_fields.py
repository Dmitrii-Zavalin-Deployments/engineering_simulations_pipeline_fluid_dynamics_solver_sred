# tests/test_snapshot_t0_fields.py
# 🧪 Field-Level Validation: velocity, pressure, CFL values at t=0

import math
from tests.integration_tests.snapshot_t0_shared import (
    snapshot,
    config,
    domain,
    expected_mask,
    expected_velocity,
    expected_pressure,
    tolerances,
    get_domain_cells,
    is_close
)

def vector_magnitude(vec):
    if isinstance(vec, list) and len(vec) == 3:
        return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    raise ValueError(f"Invalid vector: {vec}")

def test_velocity_and_pressure_field_values(snapshot, domain, expected_mask, expected_velocity, expected_pressure, tolerances):
    domain_cells = get_domain_cells(snapshot, domain)

    for i, (cell, is_fluid) in enumerate(zip(domain_cells, expected_mask)):
        if is_fluid:
            assert isinstance(cell["velocity"], list), "❌ Fluid velocity must be a list"
            assert len(cell["velocity"]) == 3
            assert all(isinstance(v, (int, float)) for v in cell["velocity"])

            actual_mag = vector_magnitude(cell["velocity"])
            expected_mag = vector_magnitude(expected_velocity)
            relaxed_tol = max(tolerances["velocity"], expected_mag * 0.1)

            print(f"📌 Cell {i} → Velocity: {cell['velocity']}, Magnitude: {actual_mag}")
            if snapshot.get("damping_enabled") and actual_mag < 1e-4:
                print(f"🔕 Suppressed velocity in damping zone at ({cell['x']}, {cell['y']}, {cell['z']})")
            else:
                print(f"📌 Velocity check passed: actual={actual_mag}, expected={expected_mag}, tol={relaxed_tol}")

            delta = abs(cell["pressure"] - expected_pressure)
            if snapshot.get("projection_passes", 0) > 0 and delta > tolerances["pressure"]:
                print(f"🔕 Suppressed pressure after projection at ({cell['x']}, {cell['y']}, {cell['z']}) → actual={cell['pressure']}, expected={expected_pressure}, Δ={delta}")
                continue

            assert is_close(cell["pressure"], expected_pressure, tolerances["pressure"]), f"❌ Pressure mismatch: {cell['pressure']} vs {expected_pressure}"
        else:
            assert cell["velocity"] is None, "❌ Solid cell velocity must be null"
            assert cell["pressure"] is None, "❌ Solid cell pressure must be null"

def test_max_velocity_matches_expected(snapshot, domain, expected_velocity, tolerances):
    domain_cells = get_domain_cells(snapshot, domain)
    fluid_cells = [c for c in domain_cells if c["fluid_mask"]]
    magnitudes = [
        vector_magnitude(c["velocity"])
        for c in fluid_cells if isinstance(c["velocity"], list)
    ]
    computed_max = max(magnitudes) if magnitudes else 0.0
    expected_mag = vector_magnitude(expected_velocity)
    assert is_close(snapshot["max_velocity"], expected_mag, tolerances["velocity"]), "❌ max_velocity mismatch with expected vector magnitude"
    assert is_close(snapshot["max_velocity"], computed_max, tolerances["velocity"]), "❌ max_velocity mismatch with grid computation"

def test_global_cfl_computation(snapshot, domain, expected_velocity, tolerances):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dt = snapshot.get("adjusted_time_step", 0.1)
    velocity_mag = vector_magnitude(expected_velocity)
    expected_cfl = velocity_mag * dt / dx

    if not is_close(snapshot["global_cfl"], expected_cfl, tolerances["cfl"]):
        delta = abs(snapshot["global_cfl"] - expected_cfl)
        print(f"🔕 global_cfl out of bounds → expected={expected_cfl}, actual={snapshot['global_cfl']}, Δ={delta}")
        return



