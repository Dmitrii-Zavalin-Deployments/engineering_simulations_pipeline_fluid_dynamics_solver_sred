# File: src/scoring/mutation_suppression_report.py

import json
import os
import logging

logger = logging.getLogger(__name__)

def export_suppression_report(step_index: int, suppressed_cells: list, output_dir: str):
    """
    Generate a JSON report of mutation-suppressed fluid cells adjacent to ghosts.

    Parameters:
    - step_index: integer timestep
    - suppressed_cells: list of dicts with keys 'cell': [x, y, z], 'reason': str
    - output_dir: base path where the report will be saved
    """
    if not suppressed_cells:
        logger.debug(f"[report] No suppressed mutations for step {step_index}")
        return

    report = {
        "step": step_index,
        "suppressed": suppressed_cells
    }

    os.makedirs(output_dir, exist_ok=True)
    filename = f"mutation_suppression_step_{step_index:04d}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[report] Suppression report exported → {filepath}")



