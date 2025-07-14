# src/physics/velocity_projection.py
# 💨 Velocity Projection — adjusts fluid velocity using pressure gradient ∇p

from typing import List
from src.grid_modules.cell import Cell

def apply_pressure_velocity_projection(grid, config: dict) -> List[Cell]:
    """
    Projects velocity by subtracting pressure gradient using central difference approximation.
    Enforces incompressibility after pressure solve.

    Args:
        grid (List[Cell] or Cell): Simulation grid or single cell with updated pressures
        config (dict): Full simulation config including domain resolution

    Returns:
        List[Cell]: Grid with updated velocity fields
    """
    if isinstance(grid, Cell):
        grid = [grid]

    domain = config.get("domain_definition", {})
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    spacing = {
        "x": dx,
        "y": dy,
        "z": dz
    }

    # 🗺️ Create coordinate-indexed lookup
    pressure_map = {(c.x, c.y, c.z): c.pressure for c in grid if c.pressure is not None}
    velocity_map = {(c.x, c.y, c.z): c.velocity for c in grid if c.velocity is not None}

    updated = []
    for cell in grid:
        coord = (cell.x, cell.y, cell.z)
        if not cell.fluid_mask or coord not in velocity_map:
            updated.append(cell)
            continue

        grad = [0.0, 0.0, 0.0]
        offsets = [("x", dx, (1, 0, 0)), ("y", dy, (0, 1, 0)), ("z", dz, (0, 0, 1))]

        for i, (_, h, delta) in enumerate(offsets):
            plus = (cell.x + delta[0]*h, cell.y + delta[1]*h, cell.z + delta[2]*h)
            minus = (cell.x - delta[0]*h, cell.y - delta[1]*h, cell.z - delta[2]*h)
            p_plus = pressure_map.get(plus)
            p_minus = pressure_map.get(minus)
            if p_plus is not None and p_minus is not None:
                grad[i] = (p_plus - p_minus) / (2.0 * h)

        projected_velocity = [v - g for v, g in zip(velocity_map[coord], grad)]

        updated.append(Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=projected_velocity,
            pressure=cell.pressure,
            fluid_mask=True
        ))

    return updated



