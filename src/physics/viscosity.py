# src/physics/viscosity.py
# 💧 Viscous Diffusion Module — applies μ∇²u smoothing to velocity field with mutation diagnostics

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple
from src.physics.advection_methods.helpers import vector_add, vector_scale, copy_cell
import math

def apply_viscous_terms(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Applies Laplacian-based smoothing to velocity field using 6-point stencil.

    Roadmap Alignment:
    Governing Equation:
        Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u + F

    This module enforces:
    - μ∇²u → viscous diffusion via Laplacian stencil
    - ∂u/∂t → explicit Euler update

    Purpose:
    - Dampens high-frequency velocity fluctuations
    - Anchors viscous term in momentum solver
    - Supports reflex diagnostics and mutation traceability

    Strategy:
    1. For each fluid cell, compute average velocity of 6 neighbors
    2. Subtract current velocity to get Laplacian term
    3. Apply explicit Euler update: u_new = u_old + μ∇²u · dt

    Args:
        grid (List[Cell]): Current grid with velocity and pressure
        dt (float): Time step duration
        config (dict): Fluid properties and simulation config

    Returns:
        List[Cell]: Grid with viscosity-adjusted velocities (fluid cells only)
    """
    viscosity = config.get("fluid_properties", {}).get("viscosity", 0.0)
    domain = config.get("domain_definition", {})

    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / domain.get("nx", 1)
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / domain.get("ny", 1)
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / domain.get("nz", 1)

    grid_lookup: Dict[Tuple[float, float, float], Cell] = {
        (cell.x, cell.y, cell.z): cell for cell in grid
    }

    updated = []
    mutation_count = 0

    for cell in grid:
        if not cell.fluid_mask or not isinstance(cell.velocity, list):
            updated.append(copy_cell(cell))
            continue

        # 🧭 6-point stencil offsets for Laplacian
        neighbor_offsets = [
            (dx, 0.0, 0.0), (-dx, 0.0, 0.0),
            (0.0, dy, 0.0), (0.0, -dy, 0.0),
            (0.0, 0.0, dz), (0.0, 0.0, -dz),
        ]

        neighbors = []
        for offset in neighbor_offsets:
            pos = (cell.x + offset[0], cell.y + offset[1], cell.z + offset[2])
            neighbor = grid_lookup.get(pos)
            if neighbor and neighbor.fluid_mask and isinstance(neighbor.velocity, list):
                neighbors.append(neighbor.velocity)

        if not neighbors:
            updated.append(copy_cell(cell))
            continue

        # 🧮 Compute average neighbor velocity
        avg_velocity = [0.0, 0.0, 0.0]
        for v in neighbors:
            for i in range(3):
                avg_velocity[i] += v[i]
        for i in range(3):
            avg_velocity[i] /= len(neighbors)

        # ∇²u ≈ avg_neighbors - u
        laplacian_term = [avg_velocity[i] - cell.velocity[i] for i in range(3)]

        # ⏱️ Euler update: u_new = u_old + μ∇²u · dt
        viscosity_update = vector_scale(laplacian_term, viscosity * dt)
        new_velocity = vector_add(cell.velocity, viscosity_update)

        # 📊 Mutation diagnostics
        delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_velocity, cell.velocity)))
        if delta > 1e-8:
            mutation_count += 1

        updated.append(copy_cell(cell, velocity=new_velocity))

    print(f"💧 Viscosity applied to {len(updated)} cells → {mutation_count} velocity mutations")

    return updated



