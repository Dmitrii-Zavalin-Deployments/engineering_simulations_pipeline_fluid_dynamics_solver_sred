# src/ci/reflex_log_score.py
# 📊 Reflex Log Score — CI log parser and simulation summary evaluator

import os
from src.metrics.reflex_score_evaluator import evaluate_reflex_score

MARKERS = {
    "Pressure updated @": "pressure_mutation",
    "Divergence stats (before projection):": "divergence_tracking_pre",
    "Divergence stats (after projection):": "divergence_tracking_post",
    "Influence skipped: matched fields": "ghost_suppression",
    "Mutation pathway recorded →": "mutation_pathway",
    "Pressure delta map saved →": "pressure_delta",
    "Step summary": "step_summary_detected",
    "[AUDIT] Step": "audit_report_triggered",
    "[COMPACTOR] ✅ Compacted snapshot saved": "snapshot_compacted",
    # ✅ Patch: fallback scoring detection
    "Mutation near ghost but tagging suppressed → soft fallback applied": "fallback_applied"
}

def score_reflex_log_text(log_text: str) -> dict:
    """
    Scans CI logs for key diagnostic markers and computes a simple reflex score.

    Parameters:
    - log_text: string containing raw build logs

    Returns:
    - dict with reflex score summary
    """
    matched = [label for marker, label in MARKERS.items() if marker in log_text]
    found = len(matched)
    total = len(MARKERS)
    return {
        "reflex_score": f"{found} / {total}",
        "markers_matched": matched
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



