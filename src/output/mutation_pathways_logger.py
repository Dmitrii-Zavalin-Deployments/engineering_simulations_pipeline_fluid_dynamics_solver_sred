# src/output/mutation_pathways_logger.py
# 🧠 Mutation Pathways Logger — tracks pressure mutation causality per timestep

import os
import json

def log_mutation_pathway(
    step_index: int,
    pressure_mutated: bool,
    triggered_by: list,
    output_folder: str = "data/testing-input-output/navier_stokes_output"
):
    """
    Appends mutation trace entry to mutation_pathways_log.json.

    Args:
        step_index (int): Current simulation step index
        pressure_mutated (bool): True if pressure field was updated
        triggered_by (list): List of strings identifying mutation causes
        output_folder (str): Location to write JSON log file
    """
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "mutation_pathways_log.json")

    entry = {
        "step_index": step_index,
        "pressure_mutated": pressure_mutated,
        "triggered_by": triggered_by
    }

    # Read or initialize
    try:
        with open(log_path, "r") as f:
            log = json.load(f)
            if not isinstance(log, list):
                log = []
    except Exception:
        log = []

    log.append(entry)

    # Write updated log
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[DEBUG] 🔄 Mutation pathway recorded → {log_path}")



