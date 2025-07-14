# src/physics/velocity_projection.py
# 💨 Velocity Projection — adjusts fluid velocity using pressure gradient ∇p

from typing import List, Tuple
from src.grid_modules.cell import Cell

def apply_pressure_velocity_projection(grid: List[Cell], config: dict) -> List[Cell]:
    """
    Projects velocity by subtracting pressure gradient using central difference approximation.
    Enforces incompressibility after pressure solve.

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

    spacing = {
        "x": dx,
        "y": dy,
        "z": dz
    }

    # 🗺️ Create coordinate-indexed lookup
    pressure_map = {(cell.x, cell.y, cell.z): cell.pressure for cell in grid if cell.pressure is not None}
    velocity_map = {(cell.x, cell.y, cell.z): cell.velocity for cell in grid if cell.velocity is not None}

    coord_set = set(pressure_map.keys())

    updated = []
    for cell in grid:
        coord = (cell.x, cell.y, cell.z)
        if not cell.fluid_mask or coord not in velocity_map:
            updated.append(cell)
            continue

        # Compute central gradient for each axis
        grad = [0.0, 0.0, 0.0]
        offsets = [("x", dx, (1, 0, 0)), ("y", dy, (0, 1, 0)), ("z", dz, (0, 0, 1))]

        for i, (axis, h, delta) in enumerate(offsets):
            p_plus = pressure_map.get((cell.x + delta[0]*h, cell.y + delta[1]*h, cell.z + delta[2]*h))
            p_minus = pressure_map.get((cell.x - delta[0]*h, cell.y - delta[1]*h, cell.z - delta[2]*h))
            if p_plus is not None and p_minus is not None:
                grad[i] = (p_plus - p_minus) / (2.0 * h)

        original_velocity = velocity_map[coord]
        projected_velocity = [v - g for v, g in zip(original_velocity, grad)]

        updated_cell = Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=projected_velocity,
            pressure=cell.pressure,
            fluid_mask=True
        )
        updated.append(updated_cell)

    return updated



