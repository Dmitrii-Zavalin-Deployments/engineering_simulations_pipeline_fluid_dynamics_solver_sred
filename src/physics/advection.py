# src/physics/advection.py
# 🌀 Advection Operator — computes nonlinear transport term u · ∇u for momentum enforcement

from typing import List
from src.grid_modules.cell import Cell

def compute_advection(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Computes velocity advection using central difference approximation.

    Roadmap Alignment:
    Governing Equation:
        Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u + F

    This module enforces:
    - u · ∇u → nonlinear convective transport
    - ∂u/∂t → explicit Euler update

    Purpose:
    - Captures directional momentum transfer across fluid cells
    - Anchors convective term in momentum solver
    - Supports reflex diagnostics and mutation traceability

    Strategy:
    1. For each fluid cell, compute spatial derivatives of velocity components
    2. Multiply by local velocity to get nonlinear transport
    3. Apply explicit Euler update: u_new = u_old - dt * (u · ∇u)

    Args:
        grid (List[Cell]): Simulation grid with velocity fields
        dt (float): Time step size
        config (dict): Full simulation config including domain resolution

    Returns:
        List[Cell]: Grid with advected velocity fields
    """
    domain = config.get("domain_definition", {})
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    # 🗺️ Build spatial lookup for velocity field
    velocity_map = {(c.x, c.y, c.z): c.velocity for c in grid if c.velocity is not None}

    advected = []
    for cell in grid:
        coord = (cell.x, cell.y, cell.z)

        # 🚫 Skip non-fluid or malformed cells
        if not getattr(cell, "fluid_mask", False) or coord not in velocity_map:
            advected.append(cell)
            continue

        u = velocity_map[coord]
        grad_u = [None, None, None]

        # 🧮 Central difference for each velocity component ∂u/∂x, ∂u/∂y, ∂u/∂z
        for i, (h, delta) in enumerate([(dx, (1, 0, 0)), (dy, (0, 1, 0)), (dz, (0, 0, 1))]):
            plus = (cell.x + delta[0]*h, cell.y + delta[1]*h, cell.z + delta[2]*h)
            minus = (cell.x - delta[0]*h, cell.y - delta[1]*h, cell.z - delta[2]*h)
            v_plus = velocity_map.get(plus)
            v_minus = velocity_map.get(minus)

            if v_plus and v_minus:
                grad_u[i] = [(vp - vm) / (2.0 * h) for vp, vm in zip(v_plus, v_minus)]
            else:
                grad_u[i] = [0.0, 0.0, 0.0]  # fallback for boundary cells

        # 🌀 Compute nonlinear transport term u · ∇u
        transport = [sum(u[j] * grad_u[i][j] for j in range(3)) for i in range(3)]

        # ⏱️ Explicit Euler update: u_new = u_old - dt * (u · ∇u)
        new_velocity = [u[i] - dt * transport[i] for i in range(3)]

        advected.append(Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=new_velocity,
            pressure=cell.pressure,
            fluid_mask=True
        ))

    return advected



