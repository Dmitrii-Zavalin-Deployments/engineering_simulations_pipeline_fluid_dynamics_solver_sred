# src/exporters/pressure_delta_map_writer.py
# 📘 Pressure Delta Map Writer — step-level per-cell pressure change export

import json
import os

def export_pressure_delta_map(pressure_delta_map: dict, step_index: int, output_dir: str = "data/snapshots"):
    """
    Saves pressure delta information for each fluid cell to a JSON file.

    Args:
        pressure_delta_map (dict): Dict keyed by (x, y, z) tuples with pressure change info
        step_index (int): Simulation step number
        output_dir (str): Path to output directory for snapshots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    serialized = {}
    for coord, values in pressure_delta_map.items():
        x, y, z = coord
        serialized[f"({x:.2f}, {y:.2f}, {z:.2f})"] = {
            "pressure_before": round(values.get("before", 0.0), 6),
            "pressure_after": round(values.get("after", 0.0), 6),
            "delta": round(values.get("delta", 0.0), 6)
        }

    filename = f"pressure_delta_map_step_{step_index:04d}.json"
    full_path = os.path.join(output_dir, filename)

    with open(full_path, "w") as f:
        json.dump(serialized, f, indent=2)

    print(f"[EXPORT] ✅ Pressure delta map saved → {full_path}")



