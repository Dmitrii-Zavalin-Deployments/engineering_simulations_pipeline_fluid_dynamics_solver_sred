# tests/metrics/test_overflow_monitor.py
# ✅ Validation suite for src/metrics/overflow_monitor.py

import pytest
from src.metrics.overflow_monitor import detect_overflow
from src.grid_modules.cell import Cell
import math

def make_cell(x, y, z, velocity=None, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=0.0,
        fluid_mask=fluid_mask
    )

def test_detect_overflow_returns_false_for_empty_grid():
    assert detect_overflow([]) is False
    assert detect_overflow(None) is False

def test_detect_overflow_excludes_non_fluid_cells():
    cell_solid = make_cell(0.0, 0.0, 0.0, velocity=[100.0, 0.0, 0.0], fluid_mask=False)
    assert detect_overflow([cell_solid]) is False

def test_detect_overflow_detects_single_spike():
    cell_normal = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 2.0, 3.0])
    cell_spike = make_cell(1.0, 1.0, 1.0, velocity=[10.0, 10.0, 10.0])
    assert detect_overflow([cell_normal, cell_spike]) is True

def test_detect_overflow_detects_multiple_spikes():
    spike_1 = make_cell(0.0, 0.0, 0.0, velocity=[15.0, 0.0, 0.0])
    spike_2 = make_cell(1.0, 1.0, 1.0, velocity=[0.0, 15.0, 0.0])
    spike_3 = make_cell(2.0, 2.0, 2.0, velocity=[0.0, 0.0, 15.0])
    assert detect_overflow([spike_1, spike_2, spike_3]) is True

def test_detect_overflow_returns_false_for_safe_velocities():
    safe_cells = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 2.0, 3.0]),
        make_cell(1.0, 1.0, 1.0, velocity=[4.0, 2.0, 1.0]),
        make_cell(2.0, 2.0, 2.0, velocity=[0.0, 0.0, 0.0])
    ]
    assert detect_overflow(safe_cells) is False

def test_detect_overflow_handles_malformed_velocity_gracefully():
    malformed_1 = make_cell(0.0, 0.0, 0.0, velocity=None)
    malformed_2 = make_cell(1.0, 1.0, 1.0, velocity=[1.0])  # too short
    malformed_3 = make_cell(2.0, 2.0, 2.0, velocity="fast")  # wrong type
    assert detect_overflow([malformed_1, malformed_2, malformed_3]) is False

def test_detect_overflow_threshold_boundary():
    # Magnitude = sqrt(6^2 + 8^2 + 0^2) = 10.0
    edge_cell = make_cell(0.0, 0.0, 0.0, velocity=[6.0, 8.0, 0.0])
    assert math.isclose(math.sqrt(6.0**2 + 8.0**2), 10.0)
    assert detect_overflow([edge_cell]) is False

    # Slightly above threshold
    spike_cell = make_cell(1.0, 1.0, 1.0, velocity=[6.1, 8.0, 0.0])
    assert detect_overflow([spike_cell]) is True



