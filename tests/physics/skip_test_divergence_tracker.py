# tests/physics/test_divergence_tracker.py
# ✅ Validation suite for src/physics/divergence_tracker.py

import os
import json
import tempfile
import pytest
from src.physics.divergence_tracker import compute_divergence_stats
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

def test_divergence_stats_computes_and_tags_fluid_cells():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }
    spacing = (1.0, 1.0, 1.0)
    ghost_registry = set()

    # Fluid cell with symmetric neighbors → zero divergence
    center = make_cell(1.0, 1.0, 1.0)
    x_plus = make_cell(2.0, 1.0, 1.0, velocity=[1.0, 0.0, 0.0])
    x_minus = make_cell(0.0, 1.0, 1.0, velocity=[1.0, 0.0, 0.0])
    y_plus = make_cell(1.0, 2.0, 1.0, velocity=[0.0, 2.0, 0.0])
    y_minus = make_cell(1.0, 0.0, 1.0, velocity=[0.0, 2.0, 0.0])
    z_plus = make_cell(1.0, 1.0, 2.0, velocity=[0.0, 0.0, 3.0])
    z_minus = make_cell(1.0, 1.0, 0.0, velocity=[0.0, 0.0, 3.0])

    grid = [center, x_plus, x_minus, y_plus, y_minus, z_plus, z_minus]

    with tempfile.TemporaryDirectory() as tmpdir:
        stats = compute_divergence_stats(
            grid=grid,
            spacing=spacing,
            label="test",
            step_index=1,
            output_folder=tmpdir,
            config=config,
            ghost_registry=ghost_registry
        )

        assert isinstance(stats, dict)
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert len(stats["divergence"]) == 7

        # Reflex tagging
        assert hasattr(grid[0], "divergence")
        assert grid[0].divergence == 0.0

        # File outputs
        log_path = os.path.join(tmpdir, "divergence_log.txt")
        map_path = os.path.join(tmpdir, "divergence_map_step_0001.json")

        assert os.path.exists(log_path)
        assert os.path.exists(map_path)

        with open(log_path) as f:
            log_contents = f.read()
            assert "Step 0001 | Stage: test" in log_contents

        with open(map_path) as f:
            map_data = json.load(f)
            assert f"{grid[0].x:.2f},{grid[0].y:.2f},{grid[0].z:.2f}" in map_data
            assert map_data[f"{grid[0].x:.2f},{grid[0].y:.2f},{grid[0].z:.2f}"] == 0.0

def test_divergence_stats_excludes_ghost_cells():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        }
    }
    spacing = (1.0, 1.0, 1.0)
    ghost = make_cell(0.0, 0.0, 0.0)
    fluid = make_cell(1.0, 0.0, 0.0)
    ghost_registry = {id(ghost)}

    grid = [ghost, fluid]

    with tempfile.TemporaryDirectory() as tmpdir:
        stats = compute_divergence_stats(
            grid=grid,
            spacing=spacing,
            label="ghost_test",
            step_index=2,
            output_folder=tmpdir,
            config=config,
            ghost_registry=ghost_registry
        )

        assert len(stats["divergence"]) == 1
        assert hasattr(grid[1], "divergence")
        assert not hasattr(grid[0], "divergence")



