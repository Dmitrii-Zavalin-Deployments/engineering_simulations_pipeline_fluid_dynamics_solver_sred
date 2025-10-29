# tests/metrics/test_velocity_metrics.py
# ✅ Validation suite for src/metrics/velocity_metrics.py

import pytest
import math
from src.metrics.velocity_metrics import compute_max_velocity
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=0.0,
        fluid_mask=fluid_mask
    )

def test_compute_max_velocity_returns_zero_for_empty_grid():
    assert compute_max_velocity([]) == 0.0
    assert compute_max_velocity(None) == 0.0

def test_compute_max_velocity_excludes_non_fluid_cells():
    solid = make_cell(0.0, 0.0, 0.0, velocity=[100.0, 0.0, 0.0], fluid_mask=False)
    assert compute_max_velocity([solid]) == 0.0
    assert not hasattr(solid, "overflow_triggered")

def test_compute_max_velocity_detects_max_magnitude():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[3.0, 4.0, 0.0])  # magnitude = 5.0
    c2 = make_cell(1.0, 1.0, 1.0, velocity=[6.0, 8.0, 0.0])  # magnitude = 10.0
    result = compute_max_velocity([c1, c2])
    assert result == 10.0

def test_compute_max_velocity_tags_overflow_cells():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[6.0, 8.1, 0.0])  # magnitude > 10.0
    c2 = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0])  # safe
    result = compute_max_velocity([c1, c2])
    assert result > 10.0
    assert hasattr(c1, "overflow_triggered") and c1.overflow_triggered is True
    assert c1.mutation_source == "velocity_overflow"
    assert not hasattr(c2, "overflow_triggered")

def test_compute_max_velocity_handles_malformed_velocity_gracefully():
    malformed_1 = make_cell(0.0, 0.0, 0.0, velocity=None)
    malformed_2 = make_cell(1.0, 1.0, 1.0, velocity=[1.0])  # too short
    malformed_3 = make_cell(2.0, 2.0, 2.0, velocity="fast")  # wrong type
    result = compute_max_velocity([malformed_1, malformed_2, malformed_3])
    assert result == 0.0
    for c in [malformed_1, malformed_2, malformed_3]:
        assert not hasattr(c, "overflow_triggered")

def test_compute_max_velocity_respects_custom_threshold():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[3.0, 4.0, 0.0])  # magnitude = 5.0
    c2 = make_cell(1.0, 1.0, 1.0, velocity=[6.0, 8.0, 0.0])  # magnitude = 10.0
    result = compute_max_velocity([c1, c2], overflow_threshold=9.0)
    assert result == 10.0
    assert hasattr(c2, "overflow_triggered") and c2.overflow_triggered is True



