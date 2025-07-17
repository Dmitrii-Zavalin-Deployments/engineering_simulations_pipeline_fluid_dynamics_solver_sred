# src/output/mutation_pathways_logger.py
# 🧠 Mutation Pathways Logger — tracks pressure mutation causality per timestep

import os
import json
from typing import List, Union, Optional
from src.grid_modules.cell import Cell

def serialize_cell(cell: Union[Cell, tuple], reason: Optional[str] = None) -> dict:
    if isinstance(cell, tuple):
        return {
            "x": cell[0],
            "y": cell[1],
            "z": cell[2],
            "pressure_changed": True,
            "suppression_reason": reason
        }
    return {
        "x": getattr(cell, "x", "?"),
        "y": getattr(cell, "y", "?"),
        "z": getattr(cell, "z", "?"),
        "velocity": getattr(cell, "velocity", None),
        "pressure": getattr(cell, "pressure", None),
        "fluid_mask": getattr(cell, "fluid_mask", None),
        "pressure_changed": False,
        "triggered_by": getattr(cell, "triggered_by", None),
        "influence_attempted": getattr(cell, "ghost_influence_attempted", False),
        "influence_applied": getattr(cell, "ghost_influence_applied", False),
        "suppression_reason": reason
    }

def log_mutation_pathway(
    step_index: int,
    pressure_mutated: bool,
    triggered_by: List[str],
    output_folder: str = "data/testing-input-output/navier_stokes_output",
    triggered_cells: Optional[List[Union[Cell, tuple]]] = None
):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "mutation_pathways_log.json")

    entry = {
        "step_index": step_index,
        "pressure_mutated": pressure_mutated,
        "triggered_by": triggered_by
    }

    if triggered_cells:
        entry["triggered_cells"] = [serialize_cell(cell) for cell in triggered_cells]

    try:
        with open(log_path, "r") as f:
            log = json.load(f)
            if not isinstance(log, list):
                log = []
    except Exception:
        log = []

    log.append(entry)

    try:
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
    except TypeError as e:
        print(f"[ERROR] Failed to serialize mutation pathway log: {e}")
        print(f"[DEBUG] Problematic entry that caused failure:")
        print(entry)
        raise

    print(f"Mutation pathway recorded → {log_path}")

def log_skipped_mutation(
    step_index: int,
    suppressed_cells: List[Union[Cell, tuple]],
    reason: str,
    output_folder: str = "data/testing-input-output/navier_stokes_output"
):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "mutation_pathways_log.json")

    entry = {
        "step_index": step_index,
        "pressure_mutated": False,
        "triggered_by": ["mutation suppressed"],
        "suppressed": [serialize_cell(cell, reason) for cell in suppressed_cells]
    }

    try:
        with open(log_path, "r") as f:
            log = json.load(f)
            if not isinstance(log, list):
                log = []
    except Exception:
        log = []

    log.append(entry)

    try:
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
    except TypeError as e:
        print(f"[ERROR] Failed to serialize suppressed mutation log: {e}")
        print(f"[DEBUG] Problematic entry that caused failure:")
        print(entry)
        raise

    print(f"Suppressed mutation pathway recorded → {log_path}")



