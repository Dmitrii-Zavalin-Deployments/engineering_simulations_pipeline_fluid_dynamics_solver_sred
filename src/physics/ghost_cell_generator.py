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
    # Validate top-level keys
    for key in ["domain_definition", "ghost_rules", "boundary_conditions"]:
        if key not in config:
            raise KeyError(f"Missing required '{key}' in config")

    domain = config["domain_definition"]
    ghost_rules = config["ghost_rules"]
    boundary_conditions = config["boundary_conditions"]

    # Validate domain keys
    required_domain_keys = ["nx", "ny", "nz", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]
    for key in required_domain_keys:
        if key not in domain:
            raise KeyError(f"Missing required '{key}' in domain_definition")

    nx = domain["nx"]
    ny = domain["ny"]
    nz = domain["nz"]
    dx = (domain["max_x"] - domain["min_x"]) / nx
    dy = (domain["max_y"] - domain["min_y"]) / ny
    dz = (domain["max_z"] - domain["min_z"]) / nz
    x_min, x_max = domain["min_x"], domain["max_x"]
    y_min, y_max = domain["min_y"], domain["max_y"]
    z_min, z_max = domain["min_z"], domain["max_z"]

    # Validate ghost_rules keys
    for key in ["boundary_faces", "face_types", "default_type"]:
        if key not in ghost_rules:
            raise KeyError(f"Missing required '{key}' in ghost_rules")

    boundary_faces = ghost_rules["boundary_faces"]
    face_types = ghost_rules["face_types"]
    default_type = ghost_rules["default_type"]

    if debug:
        print("[GHOST] 📘 Ghost rule config:")
        print(f"[GHOST]    Boundary faces: {boundary_faces}")
        print(f"[GHOST]    Default type: {default_type}")
        print(f"[GHOST]    Face Types: {face_types}")

    # Build lookup from boundary_conditions
    face_bc_map = {}
    for bc in boundary_conditions:
        for face in bc["apply_faces"]:
            face_bc_map[face] = {
                "velocity": bc["velocity"] if "velocity" in bc else None,
                "pressure": bc["pressure"] if "pressure" in bc else None,
                "apply_to": bc["apply_to"],
                "type": bc["type"],
                "role": bc.get("role", "unknown")
            }

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
        if "pressure" in bc["apply_to"]:
            if bc["type"] == "dirichlet":
                if bc["pressure"] is None:
                    raise ValueError(f"Missing pressure for face '{face}' with Dirichlet enforcement")
                pressure = bc["pressure"]

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

        if "x_min" in boundary_faces and abs(abs(x - x_min) - 0.5 * dx) <= EPSILON:
            face_type = face_types["x_min"]
            add_ghost(x - dx, y, z, "x_min", (x, y, z), cell, face_type)
        if "x_max" in boundary_faces and abs(abs(x - x_max) - 0.5 * dx) <= EPSILON:
            face_type = face_types["x_max"]
            add_ghost(x + dx, y, z, "x_max", (x, y, z), cell, face_type)
        if "y_min" in boundary_faces and abs(abs(y - y_min) - 0.5 * dy) <= EPSILON:
            face_type = face_types["y_min"]
            add_ghost(x, y - dy, z, "y_min", (x, y, z), cell, face_type)
        if "y_max" in boundary_faces and abs(abs(y - y_max) - 0.5 * dy) <= EPSILON:
            face_type = face_types["y_max"]
            add_ghost(x, y + dy, z, "y_max", (x, y, z), cell, face_type)
        if "z_min" in boundary_faces and abs(abs(z - z_min) - 0.5 * dz) <= EPSILON:
            face_type = face_types["z_min"]
            add_ghost(x, y, z - dz, "z_min", (x, y, z), cell, face_type)
        if "z_max" in boundary_faces and abs(abs(z - z_max) - 0.5 * dz) <= EPSILON:
            face_type = face_types["z_max"]
            add_ghost(x, y, z + dz, "z_max", (x, y, z), cell, face_type)

    total_ghosts = len(ghost_cells)
    if debug:
        print(f"[GHOST] 📊 Ghost generation complete → total: {total_ghosts}")
        for face, count in creation_counts.items():
            if count > 0:
                print(f"[GHOST]    {face}: {count} ghosts")

    padded_grid = grid + ghost_cells
    return padded_grid, ghost_registry
