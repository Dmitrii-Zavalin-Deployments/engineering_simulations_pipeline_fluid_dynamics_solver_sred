# src/output/mutation_pathways_logger.py
# 🧠 Mutation Pathways Logger — tracks pressure mutation causality per timestep

import os
import json
from typing import List, Union, Optional
from src.grid_modules.cell import Cell

def serialize_cell(cell: Cell) -> dict:
    return {
        "x": cell.x,
        "y": cell.y,
        "z": cell.z,
        "velocity": cell.velocity,
        "pressure": cell.pressure,
        "fluid_mask": cell.fluid_mask
    }

def log_mutation_pathway(
    step_index: int,
    pressure_mutated: bool,
    triggered_by: List[str],
    output_folder: str = "data/testing-input-output/navier_stokes_output",
    triggered_cells: Optional[List[Union[Cell, tuple]]] = None
):
    """
    Appends mutation trace entry to mutation_pathways_log.json.

    Args:
        step_index (int): Current simulation step index
        pressure_mutated (bool): True if pressure field was updated
        triggered_by (List[str]): List of strings identifying mutation causes
        output_folder (str): Location to write JSON log file
        triggered_cells (Optional[List[Cell or Tuple]]): Optional list of Cell objects or coordinates
    """
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "mutation_pathways_log.json")

    entry = {
        "step_index": step_index,
        "pressure_mutated": pressure_mutated,
        "triggered_by": triggered_by
    }

    if triggered_cells:
        entry["triggered_cells"] = [
            serialize_cell(c) if isinstance(c, Cell) else {"x": c[0], "y": c[1], "z": c[2]}
            for c in triggered_cells
        ]

    # Read or initialize log safely
    try:
        with open(log_path, "r") as f:
            log = json.load(f)
            if not isinstance(log, list):
                log = []
    except Exception:
        log = []

    log.append(entry)

    # Write updated log safely
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[DEBUG] 🔄 Mutation pathway recorded → {log_path}")



