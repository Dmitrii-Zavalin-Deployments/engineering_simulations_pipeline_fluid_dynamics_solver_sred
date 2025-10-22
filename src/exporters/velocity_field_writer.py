import json
import os

# 🛠️ Toggle debug logging
DEBUG = True  # Set to True to enable verbose diagnostics

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

    if DEBUG:
        if not velocity_snapshot:
            print(f"[DEBUG] No fluid cells found for velocity export at step {step}")
        print(f"💨 Velocity field snapshot saved → {filepath}")



