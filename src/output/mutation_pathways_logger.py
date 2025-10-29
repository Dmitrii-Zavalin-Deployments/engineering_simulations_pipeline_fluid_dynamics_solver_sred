# src/output/mutation_pathways_logger.py
# 🧠 Mutation Pathways Logger — tracks pressure mutation causality and suppression diagnostics per timestep
# 📌 This module logs reflex-triggered pressure mutations and ghost influence chains.
# It supports audit-grade traceability of solver decisions and mutation propagation.
# It does NOT exclude cells based on adjacency or boundary proximity — only explicit fluid_mask=False cells are excluded upstream.

import os
import json
from typing import List, Union, Optional
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def serialize_cell(cell: Union[Cell, tuple], reason: Optional[str] = None, step_linked_from: Optional[int] = None) -> dict:
    """
    Converts a cell or coordinate tuple into a structured mutation trace entry.
    """
    base = {
        "x": getattr(cell, "x", cell[0] if isinstance(cell, tuple) else "?"),
        "y": getattr(cell, "y", cell[1] if isinstance(cell, tuple) else "?"),
        "z": getattr(cell, "z", cell[2] if isinstance(cell, tuple) else "?"),
        "pressure_changed": isinstance(cell, tuple),
        "suppression_reason": reason,
        "step_linked_from": step_linked_from
    }

    if isinstance(cell, Cell):
        base.update({
            "velocity": getattr(cell, "velocity", None),
            "pressure": getattr(cell, "pressure", None),
            "fluid_mask": getattr(cell, "fluid_mask", None),
            "triggered_by": getattr(cell, "mutation_triggered_by", None),
            "pressure_delta": getattr(cell, "pressure_delta", None),
            "influenced_by_ghost": getattr(cell, "influenced_by_ghost", False),
            "mutation_source": getattr(cell, "mutation_source", None),
            "mutation_step": getattr(cell, "mutation_step", None)
        })

    return base

def log_mutation_pathway(
    step_index: int,
    pressure_mutated: bool,
    triggered_by: List[str],
    output_folder: str = "data/testing-input-output/navier_stokes_output",
    triggered_cells: Optional[List[Union[Cell, tuple]]] = None,
    ghost_trigger_chain: Optional[List[int]] = None
):
    """
    Logs mutation pathway for a given timestep.
    """
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "mutation_pathways_log.json")

    entry = {
        "step_index": step_index,
        "pressure_mutated": pressure_mutated,
        "triggered_by": triggered_by,
        "ghost_trigger_chain": ghost_trigger_chain or []
    }

    if triggered_cells:
        entry["triggered_cells"] = [
            serialize_cell(cell, step_linked_from=entry["ghost_trigger_chain"][-1] if entry["ghost_trigger_chain"] else None)
            for cell in triggered_cells
        ]
        entry["mutated_cells"] = [
            (cell.x, cell.y, cell.z)
            for cell in triggered_cells
            if hasattr(cell, "x") and hasattr(cell, "y") and hasattr(cell, "z")
        ]

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
        if debug:
            print(f"[ERROR] Failed to serialize mutation pathway log: {e}")
            print(f"[DEBUG] Problematic entry that caused failure:")
            print(entry)
        raise

    if debug:
        print(f"[PATHWAY] Mutation pathway recorded → {log_path}")



