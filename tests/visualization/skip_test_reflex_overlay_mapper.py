import os
import pytest
import matplotlib
matplotlib.use("Agg")  # Headless backend for CI-safe rendering

from src.visualization.reflex_overlay_mapper import render_reflex_overlay

@pytest.fixture
def coords():
    return {
        "mutation": [(1.0, 1.0), (2.0, 2.0)],
        "adjacency": [(3.0, 3.0)],
        "suppression": [(4.0, 4.0)],
        "boundary": [(0.0, 0.0), (5.0, 5.0)]
    }

def test_overlay_renders_all_zones(tmp_path, coords):
    output_path = tmp_path / "reflex_overlay.png"
    render_reflex_overlay(
        step_index=1,
        reflex_score=4.5,
        mutation_coords=coords["mutation"],
        adjacency_coords=coords["adjacency"],
        suppression_coords=coords["suppression"],
        boundary_coords=coords["boundary"],
        mutation_density=0.12,
        output_path=str(output_path)
    )
    assert output_path.exists()

def test_overlay_skips_if_score_too_low(tmp_path, coords, capsys):
    output_path = tmp_path / "reflex_overlay_skipped.png"
    render_reflex_overlay(
        step_index=2,
        reflex_score=3.5,
        mutation_coords=coords["mutation"],
        adjacency_coords=coords["adjacency"],
        suppression_coords=coords["suppression"],
        boundary_coords=coords["boundary"],
        mutation_density=0.12,
        output_path=str(output_path)
    )
    assert not output_path.exists()
    output = capsys.readouterr().out
    assert "Skipping overlay" in output

def test_overlay_creates_directory(tmp_path, coords):
    nested_path = tmp_path / "nested" / "reflex_overlay.png"
    render_reflex_overlay(
        step_index=3,
        reflex_score=4.2,
        mutation_coords=coords["mutation"],
        adjacency_coords=coords["adjacency"],
        suppression_coords=coords["suppression"],
        boundary_coords=coords["boundary"],
        mutation_density=0.05,
        output_path=str(nested_path)
    )
    assert nested_path.exists()

def test_overlay_handles_missing_boundary(tmp_path, coords):
    output_path = tmp_path / "reflex_overlay_noboundary.png"
    render_reflex_overlay(
        step_index=4,
        reflex_score=4.1,
        mutation_coords=coords["mutation"],
        adjacency_coords=coords["adjacency"],
        suppression_coords=coords["suppression"],
        boundary_coords=None,
        mutation_density=0.08,
        output_path=str(output_path)
    )
    assert output_path.exists()

def test_overlay_handles_non_numeric_score(tmp_path, coords):
    output_path = tmp_path / "reflex_overlay_badscore.png"
    render_reflex_overlay(
        step_index=5,
        reflex_score="invalid",
        mutation_coords=coords["mutation"],
        adjacency_coords=coords["adjacency"],
        suppression_coords=coords["suppression"],
        boundary_coords=coords["boundary"],
        mutation_density=0.08,
        output_path=str(output_path)
    )
    assert not output_path.exists()



