# src/physics/ghost_cell_generator.py
# 🧱 Ghost Cell Generator — injects ghost padding based on boundary conditions and fluid adjacency
# 📌 This module generates ghost cells for boundary enforcement and reflex diagnostics.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or proximity — all logic is geometry-mask-driven.

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

# ✅ Floating-point tolerance for proximity checks
EPSILON = 1e-8

def generate_ghost_cells(grid: List[Cell], config: dict, debug: bool = True) -> Tuple[List[Cell], Dict[int, dict]]:
    """
    Generates ghost cells at domain boundaries based on tagged faces and no-slip enforcement.
    """
    domain = config.get("domain_definition", {})
    ghost_rules = config.get("ghost_rules", {})
    boundary_conditions = config.get("boundary_conditions", [])

    boundary_faces = ghost_rules.get("boundary_faces", [])
    face_types = ghost_rules.get("face_types", {})
    default_type = ghost_rules.get("default_type", "wall")

    if debug:
        print("[GHOST] 📘 Ghost rule config:")
        print(f"[GHOST]    Boundary faces: {boundary_faces}")
        print(f"[GHOST]    Default type: {default_type}")
        print(f"[GHOST]    Face Types: {face_types}")

    # Build lookup from boundary_conditions
    face_bc_map = {}
    for bc in boundary_conditions:
        for face in bc.get("apply_faces", []):
            face_bc_map[face] = {
                "velocity": bc.get("velocity"),
                "pressure": bc.get("pressure"),
                "apply_to": bc.get("apply_to", []),
                "type": bc.get("type", "neumann"),
                "role": bc.get("role", "unknown")
            }

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

    def add_ghost(x, y, z, face, origin, fluid_cell: Cell, face_type: str):
        bc = face_bc_map.get(face)
        if not bc:
            raise ValueError(f"No boundary condition defined for face '{face}'")

        velocity = None
        pressure = None

        if "velocity" in bc["apply_to"]:
            if bc["type"] == "dirichlet":
                if bc["velocity"] is None:
                    raise ValueError(f"Missing velocity for face '{face}' with Dirichlet enforcement")
                velocity = bc["velocity"]
            elif bc["type"] == "neumann":
                velocity = None

        if "pressure" in bc["apply_to"]:
            if bc["type"] == "dirichlet":
                if bc["pressure"] is None:
                    raise ValueError(f"Missing pressure for face '{face}' with Dirichlet enforcement")
                pressure = bc["pressure"]
            elif bc["type"] == "neumann":
                pressure = None

        ghost = Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=False)
        setattr(ghost, "ghost_face", face)
        ghost_cells.append(ghost)
        ghost_registry[id(ghost)] = {
            "face": face,
            "origin": origin,
            "coordinate": (x, y, z),
            "velocity": velocity,
            "pressure": pressure,
            "type": face_type,
            "enforcement": {
                "velocity": "velocity" in bc["apply_to"] and bc["type"] == "dirichlet",
                "pressure": "pressure" in bc["apply_to"] and bc["type"] == "dirichlet"
            }
        }
        creation_counts[face] += 1
        if debug:
            print(f"[GHOST] 🧱 Ghost @ ({ghost.x:.2f}, {ghost.y:.2f}, {ghost.z:.2f}) ← fluid @ ({fluid_cell.x:.2f}, {fluid_cell.y:.2f}, {fluid_cell.z:.2f}) → face: {face} ({face_type})")

    for cell_index, cell in enumerate(grid):
        if not cell.fluid_mask:
            continue
        x, y, z = cell.x, cell.y, cell.z
        if debug:
            print(f"[GHOST] 🔍 Evaluating fluid[{cell_index}] @ ({x:.2f}, {y:.2f}, {z:.2f})")

        if "x_min" in boundary_faces and abs(x - x_min) - 0.5 * dx <= EPSILON:
            face_type = face_types.get("x_min", default_type)
            add_ghost(x - dx, y, z, "x_min", (x, y, z), cell, face_type)
        if "x_max" in boundary_faces and abs(x - x_max) - 0.5 * dx <= EPSILON:
            face_type = face_types.get("x_max", default_type)
            add_ghost(x + dx, y, z, "x_max", (x, y, z), cell, face_type)
        if "y_min" in boundary_faces and abs(y - y_min) - 0.5 * dy <= EPSILON:
            face_type = face_types.get("y_min", default_type)
            add_ghost(x, y - dy, z, "y_min", (x, y, z), cell, face_type)
        if "y_max" in boundary_faces and abs(y - y_max) - 0.5 * dy <= EPSILON:
            face_type = face_types.get("y_max", default_type)
            add_ghost(x, y + dy, z, "y_max", (x, y, z), cell, face_type)
        if "z_min" in boundary_faces and abs(z - z_min) - 0.5 * dz <= EPSILON:
            face_type = face_types.get("z_min", default_type)
            add_ghost(x, y, z - dz, "z_min", (x, y, z), cell, face_type)
        if "z_max" in boundary_faces and abs(z - z_max) - 0.5 * dz <= EPSILON:
            face_type = face_types.get("z_max", default_type)
            add_ghost(x, y, z + dz, "z_max", (x, y, z), cell, face_type)

    total_ghosts = len(ghost_cells)
    if debug:
        print(f"[GHOST] 📊 Ghost generation complete → total: {total_ghosts}")
        for face, count in creation_counts.items():
            if count > 0:
                print(f"[GHOST]    {face}: {count} ghosts")

    padded_grid = grid + ghost_cells
    return padded_grid, ghost_registry



