# src/physics/velocity_projection.py
# 💨 Velocity Projection — adjusts fluid velocity using pressure gradient ∇P
# 📌 This module enforces incompressibility by subtracting ∇P from velocity.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from typing import List
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def apply_pressure_velocity_projection(grid: List[Cell], config: dict) -> List[Cell]:
    """
    Projects velocity field using pressure gradient subtraction.

    Roadmap Alignment:
    Governing Equation:
        Continuity: ∇ · u = 0

    Purpose:
    - After solving ∇²P = ∇ · u, we enforce incompressibility by updating velocity:
        u ← u - ∇P
    - This step completes the continuity constraint for incompressible flow
    - Ensures that the velocity field is divergence-free

    Numerical Strategy:
    - Central difference approximation of ∇P
    - Subtract gradient from velocity at each fluid cell

    Diagnostic Role:
    - Supports reflex scoring and divergence diagnostics
    - Anchors final enforcement of ∇ · u = 0

    Args:
        grid (List[Cell]): Simulation grid with updated pressures
        config (dict): Full simulation config including domain resolution

    Returns:
        List[Cell]: Grid with updated velocity fields
    """
    domain = config.get("domain_definition", {})
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    # 🗺️ Create spatial lookups
    pressure_map = {(c.x, c.y, c.z): c.pressure for c in grid if c.pressure is not None}
    velocity_map = {(c.x, c.y, c.z): c.velocity for c in grid if c.velocity is not None}

    updated = []
    for cell in grid:
        coord = (cell.x, cell.y, cell.z)

        # 🚫 Skip non-fluid or malformed cells
        if not getattr(cell, "fluid_mask", False):
            updated.append(cell)
            continue
        if coord not in velocity_map or coord not in pressure_map:
            cell.projection_skipped = True
            updated.append(cell)
            continue

        # 🧮 Compute pressure gradient ∇P using central difference
        grad = [0.0, 0.0, 0.0]
        for i, (h, delta) in enumerate([(dx, (1, 0, 0)), (dy, (0, 1, 0)), (dz, (0, 0, 1))]):
            plus = (cell.x + delta[0]*h, cell.y + delta[1]*h, cell.z + delta[2]*h)
            minus = (cell.x - delta[0]*h, cell.y - delta[1]*h, cell.z - delta[2]*h)
            p_plus = pressure_map.get(plus)
            p_minus = pressure_map.get(minus)

            if p_plus is not None and p_minus is not None:
                grad[i] = (p_plus - p_minus) / (2.0 * h)

        # 💨 Subtract pressure gradient from current velocity
        projected_velocity = [
            v - g for v, g in zip(velocity_map[coord], grad)
        ]

        updated_cell = Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=projected_velocity,
            pressure=cell.pressure,
            fluid_mask=True
        )

        if debug:
            print(f"[PROJECTION] Cell @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) → ∇P = {grad}, u_new = {projected_velocity}")

        updated.append(updated_cell)

    return updated



