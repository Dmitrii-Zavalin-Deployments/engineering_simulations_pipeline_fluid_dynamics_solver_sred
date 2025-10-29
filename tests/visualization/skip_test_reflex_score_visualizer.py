import os
import pytest
import matplotlib
matplotlib.use("Agg")  # Headless backend for CI-safe rendering

from src.visualization.reflex_score_visualizer import plot_reflex_score_evolution

@pytest.fixture
def evaluations():
    return [
        {
            "step_index": 1,
            "reflex_score": 0.85,
            "suppression_zone_count": 3,
            "adjacency_count": 5
        },
        {
            "step_index": 2,
            "reflex_score": 1.2,
            "suppression_zone_count": 6,
            "adjacency_count": 4
        },
        {
            "step_index": 3,
            "reflex_score": 0.95,
            "suppression_zone_count": 2,
            "adjacency_count": 6
        }
    ]

def test_plot_reflex_score_evolution_creates_output(tmp_path, evaluations):
    output_path = tmp_path / "reflex_score_plot.png"
    plot_reflex_score_evolution(evaluations, output_path=str(output_path))
    assert output_path.exists()

def test_plot_reflex_score_evolution_creates_directory(tmp_path, evaluations):
    nested_path = tmp_path / "plots" / "reflex_score_plot.png"
    plot_reflex_score_evolution(evaluations, output_path=str(nested_path))
    assert nested_path.exists()

def test_plot_reflex_score_evolution_handles_empty_input(tmp_path, capsys):
    output_path = tmp_path / "reflex_score_empty.png"
    plot_reflex_score_evolution([], output_path=str(output_path))
    assert not output_path.exists()
    output = capsys.readouterr().out
    assert "No evaluation data provided" in output

def test_plot_reflex_score_evolution_custom_title(tmp_path, evaluations):
    output_path = tmp_path / "reflex_score_custom.png"
    plot_reflex_score_evolution(evaluations, output_path=str(output_path), title="Custom Reflex Score Plot")
    assert output_path.exists()



