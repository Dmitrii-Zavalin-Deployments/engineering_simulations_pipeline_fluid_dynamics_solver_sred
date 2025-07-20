# ✅ Unit Test Suite — Reflex Overlay Mapper
# 📄 Full Path: tests/visualization/test_reflex_overlay_mapper.py

import pytest
import os
import matplotlib
matplotlib.use("Agg")  # Avoid GUI backend for safe testing
from tempfile import TemporaryDirectory
from src.visualization.reflex_overlay_mapper import render_reflex_overlay

def test_overlay_skipped_below_threshold(capsys):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "step_overlay.png")
        render_reflex_overlay(
            step_index=5,
            reflex_score=2.5,
            mutation_coords=[(1, 1)],
            adjacency_coords=[(2, 2)],
            suppression_coords=[(3, 3)],
            output_path=path,
            score_threshold=4.0
        )
        output = capsys.readouterr().out
        assert "Skipping overlay" in output
        assert not os.path.exists(path)

def test_overlay_rendered_successfully():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "reflex_overlay.png")
        render_reflex_overlay(
            step_index=1,
            reflex_score=5.2,
            mutation_coords=[(1.0, 1.0)],
            adjacency_coords=[(2.0, 2.0)],
            suppression_coords=[(3.0, 3.0)],
            output_path=path
        )
        assert os.path.exists(path)

def test_overlay_skipped_on_non_numeric_score(capsys):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad_score_overlay.png")
        render_reflex_overlay(
            step_index=2,
            reflex_score="invalid_score",
            mutation_coords=[],
            adjacency_coords=[],
            suppression_coords=[],
            output_path=path
        )
        output = capsys.readouterr().out
        assert "Skipping overlay" in output
        assert not os.path.exists(path)

def test_overlay_handles_empty_coords_gracefully():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty_coords.png")
        render_reflex_overlay(
            step_index=0,
            reflex_score=6.0,
            mutation_coords=[],
            adjacency_coords=[],
            suppression_coords=[],
            output_path=path
        )
        assert os.path.exists(path)

def test_overlay_writes_to_nested_directory():
    with TemporaryDirectory() as tmp:
        nested = os.path.join(tmp, "nested", "dir")
        path = os.path.join(nested, "overlay.png")
        render_reflex_overlay(
            step_index=9,
            reflex_score=10.0,
            mutation_coords=[(4.0, 4.0)],
            adjacency_coords=[],
            suppression_coords=[],
            output_path=path
        )
        assert os.path.exists(path)



