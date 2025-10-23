# src/exporters/velocity_field_writer.py
# 💨 Velocity Field Writer — exports per-cell velocity vectors for fluid cells
# 📌 This module operates on post-simulation velocity data.
# It includes only cells where fluid_mask=True.
# It does NOT exclude cells based on adjacency or boundary logic.

import json
import os

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def write_velocity_field(grid, step, output_dir="data/snapshots"):
    """
    Serialize the velocity field from the simulation grid into a structured JSON file.

    Parameters:
    - grid: List of Cell objects with velocity attributes
    - step: Integer simulation step number
    - output_dir: Directory to store the velocity field snapshot
    """
    velocity_snapshot = {}

    for cell in grid:
        if hasattr(cell, "fluid_mask") and cell.fluid_mask:
            key = f"({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f})"
            velocity_snapshot[key] = {
                "vx": cell.velocity[0],
                "vy": cell.velocity[1],
                "vz": cell.velocity[2]
            }

    os.makedirs(output_dir, exist_ok=True)
    filename = f"velocity_field_step_{step:04d}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(velocity_snapshot, f, indent=2)

    if debug:
        if not velocity_snapshot:
            print(f"[EXPORT] ⚠️ No fluid cells found for velocity export at step {step}")
        print(f"[EXPORT] 💨 Velocity field snapshot saved → {filepath}")



