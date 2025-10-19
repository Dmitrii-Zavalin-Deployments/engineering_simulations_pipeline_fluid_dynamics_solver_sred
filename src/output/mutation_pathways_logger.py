# src/output/mutation_pathways_logger.py
# 🧠 Mutation Pathways Logger — tracks pressure mutation causality and suppression diagnostics per timestep

import os
import json
from typing import List, Union, Optional
from src.grid_modules.cell import Cell

def serialize_cell(cell: Union[Cell, tuple], reason: Optional[str] = None, step_linked_from: Optional[int] = None) -> dict:
    """
    Converts a cell or coordinate tuple into a structured mutation trace entry.

    Roadmap Alignment:
    Diagnostic Output:
    - Captures mutation causality and suppression metadata
    - Supports reflex scoring, overlays, and cross-step traceability

    Args:
        cell (Cell or tuple): Cell object or (x, y, z) coordinate
        reason (str): Optional suppression reason
        step_linked_from (int): Optional step index that triggered this mutation

    Returns:
        dict: Serialized cell metadata
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
            "triggered_by": getattr(cell, "triggered_by", None),
            "influence_attempted": getattr(cell, "ghost_influence_attempted", False),
            "influence_applied": getattr(cell, "ghost_influence_applied", False)
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

    Roadmap Alignment:
    Reflex Visibility:
    - Tracks pressure mutation causality
    - Anchors reflex scoring and overlay generation
    - Supports audit-safe diagnostics and ghost-triggered step chains

    Args:
        step_index (int): Timestep index
        pressure_mutated (bool): Whether pressure field was modified
        triggered_by (List[str]): Mutation triggers (e.g. ghost influence, overflow)
        output_folder (str): Path to mutation log file
        triggered_cells (List[Cell or tuple]): Cells affected by mutation
        ghost_trigger_chain (List[int]): Prior steps that triggered mutation
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
        print(f"[ERROR] Failed to serialize mutation pathway log: {e}")
        print(f"[DEBUG] Problematic entry that caused failure:")
        print(entry)
        raise

    print(f"Mutation pathway recorded → {log_path}")

def log_skipped_mutation(
    step_index: int,
    suppressed_cells: List[Union[Cell, tuple]],
    reason: str,
    output_folder: str = "data/testing-input-output/navier_stokes_output",
    ghost_trigger_chain: Optional[List[int]] = None
):
    """
    Logs suppressed mutation pathway for a given timestep.

    Roadmap Alignment:
    Reflex Visibility:
    - Captures suppression logic and ghost adjacency fallback
    - Supports reflex scoring, overlays, and ghost-triggered step chains

    Args:
        step_index (int): Timestep index
        suppressed_cells (List[Cell or tuple]): Cells where mutation was suppressed
        reason (str): Suppression rationale
        output_folder (str): Path to mutation log file
        ghost_trigger_chain (List[int]): Prior steps that triggered mutation
    """
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "mutation_pathways_log.json")

    entry = {
        "step_index": step_index,
        "pressure_mutated": False,
        "triggered_by": ["mutation suppressed"],
        "ghost_trigger_chain": ghost_trigger_chain or [],
        "suppressed": [
            serialize_cell(cell, reason=reason, step_linked_from=ghost_trigger_chain[-1] if ghost_trigger_chain else None)
            for cell in suppressed_cells
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
    except TypeError as e:
        print(f"[ERROR] Failed to serialize suppressed mutation log: {e}")
        print(f"[DEBUG] Problematic entry that caused failure:")
        print(entry)
        raise

    print(f"Suppressed mutation pathway recorded → {log_path}")



