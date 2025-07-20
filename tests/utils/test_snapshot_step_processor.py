# ✅ Unit Test Suite — Snapshot Step Processor
# 📄 Full Path: tests/utils/test_snapshot_step_processor.py

import pytest
import os
import json
from tempfile import TemporaryDirectory
from src.utils.snapshot_step_processor import process_snapshot_step

# Dummy cell structure
class DummyCell:
    def __init__(self, x, y, z, fluid=True):
        self.x = x
        self.y = y
        self.z = z
        self.fluid_mask = fluid
        self.velocity = [1.0, 0.0, 0.0] if fluid else None
        self.pressure = 0.5 if fluid else None

def test_process_snapshot_step_runs_cleanly(monkeypatch):
    grid = [
        DummyCell(0.0, 0.0, 0.0, fluid=True),
        DummyCell(1.0, 0.0, 0.0, fluid=False)
    ]

    reflex = {
        "mutated_cells": [DummyCell(0.0, 0.0, 0.0)],
        "ghost_influence_count": 1,
        "boundary_condition_applied": True,
        "pressure_mutated": True,
        "velocity_projected": True,
        "ghost_registry": {id(grid[1]): {"coordinate": (1.0, 0.0, 0.0)}},
        "reflex_score": 0.85,
        "max_divergence": 0.05,
        "pressure_solver_invoked": True,
        "projection_skipped": False,
        "adaptive_timestep": 0.01,
        "adjacency_zones": [(0.0, 0.0)],
        "suppression_zones": [(1.0, 0.0)]
    }

    spacing = (1.0, 1.0, 1.0)
    config = {}

    with TemporaryDirectory() as tmp:
        overlays = os.path.join(tmp, "overlays")
        snapshots = os.path.join(tmp, "snapshots")
        summaries = os.path.join(tmp, "summaries")
        os.makedirs(overlays)
        os.makedirs(snapshots)
        os.makedirs(summaries)

        monkeypatch.setattr("src.output.snapshot_writer.export_influence_flags", lambda *a, **kw: None)
        monkeypatch.setattr("src.output.mutation_pathways_logger.log_mutation_pathway", lambda *a, **kw: None)
        monkeypatch.setattr("src.visualization.influence_overlay.render_influence_overlay", lambda *a, **kw: None)
        monkeypatch.setattr("src.visualization.reflex_overlay_mapper.render_reflex_overlay", lambda *a, **kw: None)
        monkeypatch.setattr("src.utils.snapshot_summary_writer.write_step_summary", lambda *a, **kw: None)
        monkeypatch.setattr("src.adaptive.grid_refiner.propose_refinement_zones", lambda *a, **kw: None)

        grid_out, snapshot = process_snapshot_step(
            step=3,
            grid=grid,
            reflex=reflex,
            spacing=spacing,
            config=config,
            expected_size=1,
            output_folder=tmp
        )

        assert isinstance(snapshot, dict)
        assert snapshot["step_index"] == 3
        assert snapshot["pressure_mutated"] is True
        assert "ghost_diagnostics" in snapshot
        assert snapshot["reflex_score"] == 0.85
        assert len(snapshot["grid"]) == 2
        assert isinstance(grid_out, list)

def test_process_handles_missing_registry(monkeypatch):
    grid = [DummyCell(0.0, 0.0, 0.0, fluid=True)]
    reflex = {
        "mutated_cells": [],
        "pressure_mutated": True,
        "velocity_projected": True,
        "reflex_score": 0.1
    }
    spacing = (1.0, 1.0, 1.0)
    config = {}

    with TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "overlays"))
        monkeypatch.setattr("src.output.snapshot_writer.export_influence_flags", lambda *a, **kw: None)
        monkeypatch.setattr("src.output.mutation_pathways_logger.log_mutation_pathway", lambda *a, **kw: None)
        monkeypatch.setattr("src.visualization.influence_overlay.render_influence_overlay", lambda *a, **kw: None)
        monkeypatch.setattr("src.visualization.reflex_overlay_mapper.render_reflex_overlay", lambda *a, **kw: None)
        monkeypatch.setattr("src.utils.snapshot_summary_writer.write_step_summary", lambda *a, **kw: None)
        monkeypatch.setattr("src.adaptive.grid_refiner.propose_refinement_zones", lambda *a, **kw: None)

        grid_out, snapshot = process_snapshot_step(
            step=1,
            grid=grid,
            reflex=reflex,
            spacing=spacing,
            config=config,
            expected_size=1,
            output_folder=tmp
        )
        assert snapshot["pressure_mutated"] is True
        assert snapshot["reflex_score"] == 0.1
        assert isinstance(snapshot["grid"], list)



