# src/reflex/reflex_pathway_logger.py
# 🧠 Reflex Pathway Logger — tracks ghost-to-mutation causality across steps for audit-safe diagnostics

import os
import json
from typing import List, Tuple, Optional, Union
from src.grid_modules.cell import Cell

def serialize_mutation_cell(cell: Union[Cell, Tuple], step_linked_from: Optional[int] = None) -> dict:
    """
    Serializes a cell or coordinate into mutation trace format.
    """
    base = {
        "x": getattr(cell, "x", cell[0] if isinstance(cell, tuple) else "?"),
        "y": getattr(cell, "y", cell[1] if isinstance(cell, tuple) else "?"),
        "z": getattr(cell, "z", cell[2] if isinstance(cell, tuple) else "?"),
        "step_linked_from": step_linked_from,
        "pressure_mutated": isinstance(cell, Cell) and getattr(cell, "pressure_mutated", False),
        "mutation_triggered_by": getattr(cell, "mutation_triggered_by", None),
        "mutation_source": getattr(cell, "mutation_source", None),
        "pressure_delta": getattr(cell, "pressure_delta", None)
    }
    return base

def log_reflex_pathway(
    step_index: int,
    mutated_cells: List[Union[Cell, Tuple]],
    ghost_trigger_chain: Optional[List[int]] = None,
    output_folder: str = "data/testing-input-output/navier_stokes_output",
    verbose: bool = False  # ✅ Optional diagnostics
):
    """
    Logs reflex mutation pathway for a given step with ghost causality trace.

    Args:
        step_index (int): Current simulation step
        mutated_cells (List): Cells affected by pressure mutation
        ghost_trigger_chain (List[int]): Steps that triggered mutation
        output_folder (str): Path to write log file
        verbose (bool): If True, prints debug info
    """
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "reflex_pathway_log.json")

    entry = {
        "step_index": step_index,
        "ghost_trigger_chain": ghost_trigger_chain or [],
        "mutated_cells": [
            serialize_mutation_cell(cell, step_linked_from=ghost_trigger_chain[-1] if ghost_trigger_chain else None)
            for cell in mutated_cells
        ]
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
    except Exception as e:
        print(f"[ERROR] Failed to write reflex pathway log: {e}")
        print(f"[DEBUG] Entry: {entry}")
        raise

    if verbose:
        print(f"[TRACE] Reflex pathway recorded for step {step_index} → {log_path}")
        print(f"[TRACE] Mutated cells: {len(mutated_cells)}")
        if ghost_trigger_chain:
            print(f"[TRACE] Ghost trigger chain: {ghost_trigger_chain}")



