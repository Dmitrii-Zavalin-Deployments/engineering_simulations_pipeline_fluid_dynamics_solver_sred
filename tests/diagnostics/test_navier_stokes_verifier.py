# tests/diagnostics/test_navier_stokes_verifier.py
# ✅ Validation suite for src/diagnostics/navier_stokes_verifier.py

import os
import json
import tempfile
from unittest.mock import patch
from src.diagnostics.navier_stokes_verifier import (
    verify_continuity,
    verify_pressure_consistency,
    verify_downgraded_cells,
    run_verification_if_triggered
)
from src.grid_modules.cell import Cell

def mock_cell(x=0, y=0, z=0, velocity=None, pressure=None, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

@patch("src.physics.divergence.compute_divergence")
def test_verify_continuity_pass(mock_divergence):
    mock_divergence.return_value = [1e-8, -2e-8, 3e-8]
    grid = [mock_cell() for _ in range(3)]
    spacing = (1.0, 1.0, 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        verify_continuity(grid, spacing, step_index=1, output_folder=tmp)
        path = os.path.join(tmp, "continuity_verification_step_0001.json")
        with open(path) as f:
            result = json.load(f)
        assert result["status"] == "PASS"
        assert result["max_divergence"] < 1e-6

@patch("src.physics.divergence.compute_divergence")
def test_verify_continuity_fail(mock_divergence):
    mock_divergence.return_value = [1e-4, -2e-4, 3e-4]
    grid = [mock_cell() for _ in range(3)]
    spacing = (1.0, 1.0, 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        verify_continuity(grid, spacing, step_index=2, output_folder=tmp)
        path = os.path.join(tmp, "continuity_verification_step_0002.json")
        with open(path) as f:
            result = json.load(f)
        assert result["status"] == "FAIL"
        assert result["max_divergence"] > 1e-6

def test_verify_pressure_consistency_pass():
    grid = [mock_cell(pressure=10100), mock_cell(pressure=99999)]
    with tempfile.TemporaryDirectory() as tmp:
        verify_pressure_consistency(grid, step_index=3, output_folder=tmp)
        path = os.path.join(tmp, "pressure_verification_step_0003.json")
        with open(path) as f:
            result = json.load(f)
        assert result["status"] == "PASS"
        assert result["flagged_cells"] == []

def test_verify_pressure_consistency_warn():
    grid = [mock_cell(pressure=1e6), mock_cell(pressure=-2e5)]
    with tempfile.TemporaryDirectory() as tmp:
        verify_pressure_consistency(grid, step_index=4, output_folder=tmp)
        path = os.path.join(tmp, "pressure_verification_step_0004.json")
        with open(path) as f:
            result = json.load(f)
        assert result["status"] == "WARN"
        assert len(result["flagged_cells"]) == 2

def test_verify_downgraded_cells_mixed():
    grid = [
        mock_cell(velocity="bad", fluid_mask=True),
        mock_cell(velocity=[0, 0, 0], fluid_mask=False),
        mock_cell(velocity=[0, 0, 0], fluid_mask=True)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        verify_downgraded_cells(grid, step_index=5, output_folder=tmp)
        path = os.path.join(tmp, "downgrade_verification_step_0005.json")
        with open(path) as f:
            result = json.load(f)
        assert result["downgraded_cell_count"] == 3
        assert any("fluid_mask=False" in cell["reasons"] for cell in result["downgraded_cells"])
        assert any("missing or malformed velocity" in cell["reasons"] for cell in result["downgraded_cells"])

def test_run_verification_if_triggered_all():
    grid = [mock_cell(pressure=1e6, velocity="bad", fluid_mask=False)]
    spacing = (1.0, 1.0, 1.0)
    flags = ["empty_divergence", "no_pressure_mutation", "downgraded_cells"]

    with tempfile.TemporaryDirectory() as tmp:
        run_verification_if_triggered(grid, spacing, step_index=6, output_folder=tmp, triggered_flags=flags)
        assert os.path.exists(os.path.join(tmp, "continuity_verification_step_0006.json"))
        assert os.path.exists(os.path.join(tmp, "pressure_verification_step_0006.json"))
        assert os.path.exists(os.path.join(tmp, "downgrade_verification_step_0006.json"))

def test_run_verification_if_triggered_none():
    grid = [mock_cell()]
    spacing = (1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        run_verification_if_triggered(grid, spacing, step_index=7, output_folder=tmp, triggered_flags=[])
        assert not os.listdir(tmp)



