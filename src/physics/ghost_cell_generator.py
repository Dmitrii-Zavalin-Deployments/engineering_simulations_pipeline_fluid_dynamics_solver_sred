# src/physics/ghost_cell_generator.py
# 🧱 Ghost Cell Generator — injects ghost padding based on boundary conditions and fluid adjacency
# 🧪 Debug-log-enabled version

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell

def generate_ghost_cells(grid: List[Cell], config: dict) -> Tuple[List[Cell], Dict[int, dict]]:
    """
    Generates ghost cells at domain boundaries based on tagged faces and no-slip enforcement.

    Roadmap Alignment:
    - Governing Equations:
        - Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u
        - Continuity: ∇ · u = 0

    Purpose:
    - Ghost cells extend the domain to enforce boundary conditions
    - Support Dirichlet (fixed value) and Neumann (zero-gradient) enforcement
    - Enable pressure coupling and velocity projection near walls and inlets

    Args:
        grid (List[Cell]): Physical simulation grid.
        config (dict): Full simulation input with domain_definition and boundary_conditions.

    Returns:
        Tuple[List[Cell], Dict[int, dict]]: Augmented grid including ghost cells, and ghost registry with metadata
    """
    domain = config.get("domain_definition", {})
    boundaries = config.get("boundary_conditions", {})
    apply_faces = boundaries.get("apply_faces", [])
    no_slip = boundaries.get("no_slip", False)
    enforced_velocity = boundaries.get("velocity", [0.0, 0.0, 0.0])
    enforced_pressure = boundaries.get("pressure", None)

    print("[DEBUG] 📘 [ghost_gen] Ghost config:")
    print(f"[DEBUG]    Apply faces: {apply_faces}")
    print(f"[DEBUG]    Enforced velocity: {enforced_velocity} (no_slip={no_slip})")
    print(f"[DEBUG]    Enforced pressure: {enforced_pressure}")

    nx = domain.get("nx", 1)
    ny = domain.get("ny", 1)
    nz = domain.get("nz", 1)
    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / nx
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / ny
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / nz
    x_min, x_max = domain.get("min_x", 0.0), domain.get("max_x", 1.0)
    y_min, y_max = domain.get("min_y", 0.0), domain.get("max_y", 1.0)
    z_min, z_max = domain.get("min_z", 0.0), domain.get("max_z", 1.0)

    ghost_cells = []
    ghost_registry = {}
    creation_counts = {face: 0 for face in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]}

    def add_ghost(x, y, z, face, origin, fluid_cell: Cell):
        vel = [0.0, 0.0, 0.0] if no_slip else enforced_velocity[:]
        pressure = enforced_pressure if isinstance(enforced_pressure, (int, float)) else None
        ghost = Cell(x=x, y=y, z=z, velocity=vel, pressure=pressure, fluid_mask=False)
        setattr(ghost, "ghost_face", face)
        ghost_cells.append(ghost)
        ghost_registry[id(ghost)] = {
            "face": face,
            "origin": origin,
            "coordinate": (x, y, z),
            "velocity": vel,
            "pressure": pressure
        }
        creation_counts[face] += 1
        print(f"[DEBUG] 🧱 Ghost created @ ({ghost.x:.2f}, {ghost.y:.2f}, {ghost.z:.2f}) ← from fluid @ ({fluid_cell.x:.2f}, {fluid_cell.y:.2f}, {fluid_cell.z:.2f}) → face: {face}")

    for cell_index, cell in enumerate(grid):
        if not cell.fluid_mask:
            continue
        x, y, z = cell.x, cell.y, cell.z
        print(f"[DEBUG] 🔍 Evaluating fluid[{cell_index}] @ ({x:.2f}, {y:.2f}, {z:.2f})")

        if "x_min" in apply_faces and abs(x - x_min) <= 0.5 * dx:
            add_ghost(x - dx, y, z, "x_min", (x, y, z), cell)
        if "x_max" in apply_faces and abs(x - x_max) <= 0.5 * dx:
            add_ghost(x + dx, y, z, "x_max", (x, y, z), cell)
        if "y_min" in apply_faces and abs(y - y_min) <= 0.5 * dy:
            add_ghost(x, y - dy, z, "y_min", (x, y, z), cell)
        if "y_max" in apply_faces and abs(y - y_max) <= 0.5 * dy:
            add_ghost(x, y + dy, z, "y_max", (x, y, z), cell)
        if "z_min" in apply_faces and abs(z - z_min) <= 0.5 * dz:
            add_ghost(x, y, z - dz, "z_min", (x, y, z), cell)
        if "z_max" in apply_faces and abs(z - z_max) <= 0.5 * dz:
            add_ghost(x, y, z + dz, "z_max", (x, y, z), cell)

    total_ghosts = len(ghost_cells)
    print(f"[DEBUG] 📊 Ghost generation complete → total: {total_ghosts}")
    for face, count in creation_counts.items():
        if count > 0:
            print(f"[DEBUG]    {face}: {count} ghosts")

    padded_grid = grid + ghost_cells
    return padded_grid, ghost_registry



