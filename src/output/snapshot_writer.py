# src/output/snapshot_writer.py
# 🟨 Influence Flags Exporter — logs fluid cells influenced by ghost boundaries per step

import os
import json
from typing import List, Tuple
from src.grid_modules.cell import Cell

def export_influence_flags(grid: List[Cell], step_index: int, output_folder: str = "data/testing-input-output/navier_stokes_output"):
    """
    Writes a JSON log of fluid cells influenced by ghost fields for ParaView or diagnostics.

    Args:
        grid (List[Cell]): Final grid state for current step
        step_index (int): Simulation step index
        output_folder (str): Folder to write influence log
    """
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "influence_flags_log.json")

    fluid_influenced = [
        {
            "step_index": step_index,
            "x": c.x,
            "y": c.y,
            "z": c.z,
            "velocity": c.velocity,
            "pressure": c.pressure
        }
        for c in grid
        if getattr(c, "fluid_mask", False) and getattr(c, "influenced_by_ghost", False)
    ]

    entry = {
        "step_index": step_index,
        "influenced_cell_count": len(fluid_influenced),
        "cells": fluid_influenced
    }

    # Append to existing log or initialize
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[DEBUG] 👣 Influence flags exported → {log_path}")



