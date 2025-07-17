# src/ci/reflex_log_score.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src.metrics.reflex_score_evaluator import evaluate_reflex_score

MARKERS = {
    "Pressure updated @": "pressure_mutation",
    "Divergence stats (after projection):": "divergence_tracking",
    "Influence skipped: matched fields": "ghost_suppression",
    "Mutation pathway recorded →": "mutation_pathway",
    "Pressure delta map saved →": "pressure_delta"
}

def score_reflex_log_text(log_text: str) -> dict:
    """
    Scans CI logs for key diagnostic markers and computes a simple reflex score.

    Parameters:
    - log_text: string containing raw build logs

    Returns:
    - dict with reflex score summary
    """
    found = sum(1 for marker in MARKERS if marker in log_text)
    total = len(MARKERS)
    return {
        "reflex_score": f"{found} / {total}",
        "markers_matched": [MARKERS[m] for m in MARKERS if m in log_text]
    }

def score_from_summary_file(summary_path: str) -> dict:
    """
    Evaluates per-step reflex score based on exported simulation summary file.

    Parameters:
    - summary_path: path to step_summary.txt from simulation output

    Returns:
    - dict containing step scores and summary statistics
    """
    return evaluate_reflex_score(summary_path)

def score_combined(log_text: str, summary_path: str) -> dict:
    """
    Computes both CI marker score and simulation summary score.

    Parameters:
    - log_text: string log text from CI build
    - summary_path: path to summary file from simulation pipeline

    Returns:
    - Combined dict with CI and simulation score results
    """
    return {
        "ci_log_score": score_reflex_log_text(log_text),
        "summary_score": score_from_summary_file(summary_path)
    }
