# tests/visualization/test_reflex_overlay_mapper.py
# ✅ Unit tests for src/visualization/reflex_overlay_mapper.py

import os
import pytest
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend for headless environments
from src.visualization.reflex_overlay_mapper import render_reflex_overlay

@pytest.fixture
def sample_coords():
    return [
        (0.5, 1.0), (1.0, 1.5), (2.0, 2.5),
        (3.0, 1.0), (4.5, 2.0)
    ]

def test_overlay_skips_if_score_below_threshold(tmp_path, sample_coords):
    out_path = tmp_path / "overlay_low_score.png"
    render_reflex_overlay(
        step_index=1,
        reflex_score=3.5,
        mutation_coords=sample_coords,
        adjacency_coords=sample_coords,
        suppression_coords=sample_coords,
        output_path=str(out_path),
        score_threshold=4.0
    )
    assert not out_path.exists()

def test_overlay_renders_if_score_ok(tmp_path, sample_coords):
    out_path = tmp_path / "overlay_pass.png"
    render_reflex_overlay(
        step_index=2,
        reflex_score=4.0,
        mutation_coords=sample_coords[:2],
        adjacency_coords=sample_coords[2:],
        suppression_coords=[],
        output_path=str(out_path),
        score_threshold=4.0
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0

def test_overlay_handles_empty_lists(tmp_path):
    out_path = tmp_path / "overlay_empty.png"
    render_reflex_overlay(
        step_index=3,
        reflex_score=4.5,
        mutation_coords=[],
        adjacency_coords=[],
        suppression_coords=[],
        output_path=str(out_path)
    )
    assert out_path.exists()

def test_overlay_creates_output_dir(tmp_path, sample_coords):
    nested_dir = tmp_path / "subdir" / "overlays"
    out_path = nested_dir / "step_overlay.png"
    render_reflex_overlay(
        step_index=4,
        reflex_score=5.0,
        mutation_coords=sample_coords,
        adjacency_coords=[],
        suppression_coords=[],
        output_path=str(out_path)
    )
    assert out_path.exists()



