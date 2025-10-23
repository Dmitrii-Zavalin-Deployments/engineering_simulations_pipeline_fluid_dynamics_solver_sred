# src/utils/divergence_tracker.py
# 📈 Divergence Tracker — logs divergence norms and exports structured logs per step
# 📌 This module computes ∇ · u for fluid cells and exports reflex-aware diagnostics.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import os
import math
from typing import List, Optional
from src.grid_modules.cell import Cell
from src.exporters.divergence_field_writer import export_divergence_map

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def compute_divergence(cell: Cell, grid: List[Cell], spacing: tuple) -> float:
    if not getattr(cell, "fluid_mask", False):
        return 0.0

    dx, dy, dz = spacing
    vx, vy, vz = cell.velocity

    def find_neighbor(x, y, z):
        return next(
            (c for c in grid if math.isclose(c.x, x) and math.isclose(c.y, y) and math.isclose(c.z, z)),
            None
        )

    def safe_velocity(c):
        return c.velocity if c and getattr(c, "fluid_mask", False) else [vx, vy, vz]

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

def compute_max_divergence(grid: List[Cell], spacing: tuple) -> float:
    divergences = [
        abs(compute_divergence(c, grid, spacing))
        for c in grid if getattr(c, "fluid_mask", False)
    ]
    return max(divergences) if divergences else 0.0

def compute_divergence_stats(
    grid: List[Cell],
    spacing: tuple,
    label: Optional[str] = None,
    step_index: Optional[int] = None,
    output_folder: Optional[str] = None,
    config: Optional[dict] = None,
    reference_divergence: Optional[dict] = None
) -> dict:
    verbosity = (config or {}).get("reflex_verbosity", "medium")
    quiet = verbosity == "low"

    divergences = {}
    for cell in grid:
        if getattr(cell, "fluid_mask", False):
            div = compute_divergence(cell, grid, spacing)
            key = (cell.x, cell.y, cell.z)
            divergences[key] = div

    values = [abs(v) for v in divergences.values()]
    if not values:
        if debug and not quiet:
            print(f"[DIVERGENCE] ⚠️ No fluid cells found for divergence tracking.")
        return {"max": 0.0, "mean": 0.0, "count": 0}

    max_div = max(values)
    mean_div = sum(values) / len(values)
    count = len(values)

    if debug and not quiet:
        print(f"[DIVERGENCE] 📈 Stats ({label or 'n/a'}) → max={max_div:.6e}, mean={mean_div:.6e}, count={count}")

    if output_folder and step_index is not None:
        os.makedirs(output_folder, exist_ok=True)
        log_path = os.path.join(output_folder, "divergence_log.txt")
        with open(log_path, "a") as f:
            f.write(
                f"Step {step_index:04d} | Stage: {label or 'n/a'} | Max: {max_div:.6e} | Mean: {mean_div:.6e} | Count: {count}\n"
            )

        if reference_divergence:
            divergence_map = {}
            for coord, post_val in divergences.items():
                pre_val = reference_divergence.get(coord, 0.0)
                divergence_map[coord] = {
                    "pre": round(pre_val, 6),
                    "post": round(post_val, 6)
                }
            export_divergence_map(divergence_map, step_index, output_folder)

    return {"max": max_div, "mean": mean_div, "count": count}



