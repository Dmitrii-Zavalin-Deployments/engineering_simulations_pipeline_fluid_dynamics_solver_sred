# src/exporters/pressure_delta_map_writer.py
# 📘 Pressure Delta Map Writer — exports per-cell pressure mutation diagnostics
# for reflex scoring and traceability
# 📌 This module operates on post-simulation pressure deltas.
# It does NOT interact with fluid_mask or geometry masking logic.
# It is NOT responsible for solver inclusion/exclusion decisions.

import json
import os
from typing import List, Dict  # ✅ Added for correct type hinting

# ✅ Centralized debug flag for GitHub Actions logging
debug = False


def export_pressure_delta_map(
    pressure_delta_map: List[Dict],
    step_index: int,
    output_dir: str = "data/snapshots"
):
    """
    Saves pressure delta information for each fluid cell to a JSON file.

    Roadmap Alignment:
    Diagnostic Output:
    - Tracks pressure mutation per cell after solving ∇²P = ∇ · u
    - Supports reflex scoring, mutation pathway logging, and snapshot overlays
    - Anchors solver visibility for CI and audit pipelines

    Args:
        pressure_delta_map (List[Dict]): List of dicts with pressure change info per cell
        step_index (int): Simulation step number
        output_dir (str): Path to output directory for snapshots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    serialized = {}
    for entry in pressure_delta_map:
        coord = f"({entry['x']:.2f}, {entry['y']:.2f}, {entry['z']:.2f})"
        serialized[coord] = {
            "pressure_before": round(entry.get("before", 0.0), 6),
            "pressure_after": round(entry.get("after", 0.0), 6),
            "delta": round(entry.get("delta", 0.0), 6)
        }

    filename = f"pressure_delta_map_step_{step_index:04d}.json"
    full_path = os.path.join(output_dir, filename)

    with open(full_path, "w") as f:
        json.dump(serialized, f, indent=2)

    if debug:
        print(
            f"[EXPORT] ✅ Pressure delta map saved → {full_path}"
        )  # ✅ Reflex anchor for CI and mutation trace
