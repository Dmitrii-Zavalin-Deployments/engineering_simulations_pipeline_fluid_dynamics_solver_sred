# src/output/snapshot_writer.py
# 🟨 Influence Flags Exporter — logs fluid cells influenced by ghost boundaries per step
# 📌 This module tracks ghost-induced influence on fluid cells for reflex diagnostics and ParaView overlays.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip cells based on adjacency, boundary proximity, or solver artifacts.

import os
import json
from typing import List, Optional
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def export_influence_flags(
    grid: List[Cell],
    step_index: int,
    output_folder: str = "data/testing-input-output/navier_stokes_output",
    config: Optional[dict] = None
):
    """
    Writes a JSON log of fluid cells influenced by ghost fields for ParaView or diagnostics.

    Args:
        grid (List[Cell]): Final grid state for current step
        step_index (int): Simulation step index
        output_folder (str): Folder to write influence log
        config (dict, optional): Reflex config for verbosity toggling
    """
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "influence_flags_log.json")

    verbosity = (config or {}).get("reflex_verbosity", "medium")
    quiet_mode = verbosity == "low"
    include_details = verbosity == "high"

    entries = []
    for cell in grid:
        if getattr(cell, "fluid_mask", False) and getattr(cell, "influenced_by_ghost", False):
            detail = {
                "step_index": step_index,
                "x": cell.x,
                "y": cell.y,
                "z": cell.z
            }
            if include_details:
                # Detect mutation types
                mutation_types = []
                if hasattr(cell, "_ghost_velocity_source"):
                    mutation_types.append("velocity")
                    detail["ghost_velocity_source"] = cell._ghost_velocity_source  # (x,y,z)
                if hasattr(cell, "_ghost_pressure_source"):
                    mutation_types.append("pressure")
                    detail["ghost_pressure_source"] = cell._ghost_pressure_source  # (x,y,z)
                detail["mutation_types"] = mutation_types
                detail["velocity"] = cell.velocity
                detail["pressure"] = cell.pressure
            entries.append(detail)

    log_entry = {
        "step_index": step_index,
        "influenced_cell_count": len(entries),
        "cells": entries
    }

    # Append to existing log file
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                existing = json.load(f)
        except:
            existing = []
    else:
        existing = []

    existing.append(log_entry)

    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    if not quiet_mode and debug:
        print(f"[INFLUENCE] 👣 Influence flags exported → {log_path}")
        print(f"[INFLUENCE] ✏️ Step {step_index}: {len(entries)} fluid cells tagged as influenced by ghosts")



