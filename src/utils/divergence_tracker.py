# src/utils/divergence_tracker.py
# 📈 Divergence Tracker — logs divergence norms and exports structured logs per step

import os
import math
from typing import List, Optional
from src.grid_modules.cell import Cell

def compute_divergence(cell: Cell, grid: List[Cell], spacing: tuple) -> float:
    """
    Computes central differenced divergence for a single fluid cell.

    Args:
        cell (Cell): The cell to evaluate.
        grid (List[Cell]): Full grid for neighbor lookup.
        spacing (tuple): (dx, dy, dz) grid spacing.

    Returns:
        float: ∇·u divergence at cell
    """
    if not getattr(cell, "fluid_mask", False):
        return 0.0

    dx, dy, dz = spacing
    vx, vy, vz = cell.velocity

    def find_neighbor(x, y, z):
        return next((c for c in grid if math.isclose(c.x, x) and math.isclose(c.y, y) and math.isclose(c.z, z)), None)

    def safe_velocity(c): return c.velocity if c and getattr(c, "fluid_mask", False) else [vx, vy, vz]

    neighbor_x_plus = find_neighbor(cell.x + dx, cell.y, cell.z)
    neighbor_x_minus = find_neighbor(cell.x - dx, cell.y, cell.z)
    neighbor_y_plus = find_neighbor(cell.x, cell.y + dy, cell.z)
    neighbor_y_minus = find_neighbor(cell.x, cell.y - dy, cell.z)
    neighbor_z_plus = find_neighbor(cell.x, cell.y, cell.z + dz)
    neighbor_z_minus = find_neighbor(cell.x, cell.y, cell.z - dz)

    du_dx = (safe_velocity(neighbor_x_plus)[0] - safe_velocity(neighbor_x_minus)[0]) / (2 * dx)
    dv_dy = (safe_velocity(neighbor_y_plus)[1] - safe_velocity(neighbor_y_minus)[1]) / (2 * dy)
    dw_dz = (safe_velocity(neighbor_z_plus)[2] - safe_velocity(neighbor_z_minus)[2]) / (2 * dz)

    return du_dx + dv_dy + dw_dz

def compute_divergence_stats(
    grid: List[Cell],
    spacing: tuple,
    label: Optional[str] = None,
    step_index: Optional[int] = None,
    output_folder: Optional[str] = None
) -> dict:
    """
    Computes summary divergence statistics across fluid cells and optionally writes to a log file.

    Args:
        grid (List[Cell]): Grid after advection or projection.
        spacing (tuple): Grid spacing.
        label (str, optional): Description for debug and logging.
        step_index (int, optional): Simulation step number.
        output_folder (str, optional): Folder to write divergence_log.txt into.

    Returns:
        dict: Summary stats including max, mean, and count.
    """
    divergences = [
        abs(compute_divergence(c, grid, spacing))
        for c in grid if getattr(c, "fluid_mask", False)
    ]

    if not divergences:
        print(f"[DEBUG] ⚠️ No fluid cells found for divergence tracking.")
        return {"max": 0.0, "mean": 0.0, "count": 0}

    max_div = max(divergences)
    mean_div = sum(divergences) / len(divergences)
    count = len(divergences)

    # 💬 Console log
    if label:
        print(f"[DEBUG] 📈 Divergence stats ({label}):")
    print(f"   Max divergence: {max_div:.6e}")
    print(f"   Mean divergence: {mean_div:.6e}")
    print(f"   Cells evaluated: {count}")

    # 📝 File log (optional)
    if output_folder and step_index is not None:
        os.makedirs(output_folder, exist_ok=True)
        log_path = os.path.join(output_folder, "divergence_log.txt")
        with open(log_path, "a") as f:
            f.write(
                f"Step {step_index:04d} | Stage: {label or 'n/a'} | Max: {max_div:.6e} | Mean: {mean_div:.6e} | Count: {count}\n"
            )

    return {"max": max_div, "mean": mean_div, "count": count}

def dump_divergence_map(grid: List[Cell], spacing: tuple, path: Optional[str] = None):
    """
    Optionally export per-cell divergence values to JSON for visualization.

    Args:
        grid (List[Cell]): Full simulation grid.
        spacing (tuple): Grid spacing.
        path (str, optional): Output file path.

    Returns:
        List[dict]: List of divergence records per cell.
    """
    data = []
    for cell in grid:
        if getattr(cell, "fluid_mask", False):
            div = compute_divergence(cell, grid, spacing)
            data.append({
                "x": cell.x,
                "y": cell.y,
                "z": cell.z,
                "divergence": div
            })
    if path:
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DEBUG] 📤 Divergence map written → {path}")
    return data



