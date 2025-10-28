# src/exporters/divergence_field_writer.py
# 📘 Divergence Field Writer — exports per-cell divergence before and after
# projection
# 📌 This module operates on post-simulation divergence diagnostics.
# It does NOT interact with fluid_mask or geometry masking logic.
# It is NOT responsible for solver inclusion/exclusion decisions.

import json
import os

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def export_divergence_map(
    divergence_map: dict,
    step_index: int,
    output_dir: str = "data/snapshots"
):
    """
    Exports per-cell divergence values before and after projection.

    Args:
        divergence_map (dict): Dict keyed by (x, y, z) tuples with divergence values
        step_index (int): Simulation step number
        output_dir (str): Output directory path
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    serialized = {}
    for coord, values in divergence_map.items():
        x, y, z = coord
        serialized[f"({x:.2f}, {y:.2f}, {z:.2f})"] = {
            "divergence_before": round(values.get("pre", 0.0), 6),
            "divergence_after": round(values.get("post", 0.0), 6)
        }

    filename = f"divergence_map_step_{step_index:04d}.json"
    full_path = os.path.join(output_dir, filename)

    with open(full_path, "w") as f:
        json.dump(serialized, f, indent=2)

    if debug:
        print(
            f"[EXPORT] ✅ Divergence map saved → {full_path}"
        )
