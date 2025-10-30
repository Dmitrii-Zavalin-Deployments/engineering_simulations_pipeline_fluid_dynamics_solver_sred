# src/initialization/fluid_mask_initializer.py
# 🧬 Fluid Mask Initializer — constructs simulation grid and assigns fluid, solid, and ghost masks with reflex-aware tagging
# 📌 This module enforces geometry-mask-driven inclusion logic.
# Only ghost-boundary cells are marked fluid_mask=False.
# All other cells are initialized as fluid_mask=True.

from typing import List, Dict
from src.grid_modules.cell import Cell
from src.config.config_validator import validate_config

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def extract_domain_bounds(domain: Dict) -> tuple:
    """
    Extracts resolution and physical bounds from the domain definition.

    Raises:
        ValueError: If any required domain key is missing.

    Returns:
        Tuple of (nx, ny, nz, x_min, x_max, y_min, y_max, z_min, z_max)
    """
    required_keys = [
        "nx", "ny", "nz",
        "x_min", "x_max",
        "y_min", "y_max",
        "z_min", "z_max"
    ]
    missing = [key for key in required_keys if key not in domain]
    if missing:
        raise ValueError(f"Missing required domain keys: {', '.join(missing)}")

    return (
        domain["nx"], domain["ny"], domain["nz"],
        domain["x_min"], domain["x_max"],
        domain["y_min"], domain["y_max"],
        domain["z_min"], domain["z_max"]
    )

def get_boundary_condition(face: str, config: Dict) -> Dict:
    """
    Returns the boundary condition dict for a given face, if any.
    """
    for bc in config.get("boundary_conditions", []):
        if face in bc.get("apply_faces", []):
            return bc
    return {}

def initialize_masks(grid: List[Cell], config: Dict) -> List[Cell]:
    """
    Applies reflex-aware fluid/ghost tagging to a raw grid.

    Args:
        grid (List[Cell]): Raw simulation grid
        config (Dict): Domain and boundary configuration

    Returns:
        List[Cell]: Grid with updated mask and reflex metadata
    """
    validate_config(config)

    domain = config["domain_definition"]
    ghost_rules = config["ghost_rules"]
    boundary_faces = ghost_rules["boundary_faces"]
    ghost_type_default = ghost_rules["default_type"]

    nx, ny, nz, x_min, x_max, y_min, y_max, z_min, z_max = extract_domain_bounds(domain)

    initialized = []

    for cell in grid:
        x, y, z = cell.x, cell.y, cell.z
        fluid = True
        ghost_face = None
        ghost_type = ghost_type_default
        boundary_tag = None
        velocity = None
        pressure = None

        # ✅ Canonical ghost face detection
        if x <= x_min:
            ghost_face = "x_min"
        elif x >= x_max:
            ghost_face = "x_max"
        elif y <= y_min:
            ghost_face = "y_min"
        elif y >= y_max:
            ghost_face = "y_max"
        elif z <= z_min:
            ghost_face = "z_min"
        elif z >= z_max:
            ghost_face = "z_max"

        if ghost_face:
            fluid = False
            boundary_tag = ghost_face
            ghost_type = ghost_rules.get("face_types", {}).get(ghost_face, ghost_type_default)

            # ✅ Apply boundary condition if Dirichlet
            bc = get_boundary_condition(ghost_face, config)
            if bc.get("type") == "dirichlet":
                if "velocity" in bc.get("apply_to", []):
                    velocity = bc.get("velocity", None)
                if "pressure" in bc.get("apply_to", []):
                    pressure = bc.get("pressure", None)

        initialized_cell = Cell(
            x=x,
            y=y,
            z=z,
            velocity=velocity,
            pressure=pressure,
            fluid_mask=fluid
        )

        if not fluid:
            initialized_cell.ghost_face = ghost_face
            initialized_cell.boundary_tag = boundary_tag
            initialized_cell.ghost_type = ghost_type
            initialized_cell.ghost_source_step = config.get("step_index", None)
            initialized_cell.was_enforced = ghost_face in boundary_faces
            initialized_cell.originated_from_boundary = True
            initialized_cell.mutation_triggered_by = "boundary_enforcement"

            if debug:
                print(f"[MASK] Ghost cell @ ({x:.2f}, {y:.2f}, {z:.2f}) → face={ghost_face}, type={ghost_type}")

        initialized.append(initialized_cell)

    return initialized

def build_simulation_grid(config: Dict) -> List[Cell]:
    """
    Constructs the simulation grid and applies fluid/ghost masks.

    Args:
        config (Dict): Domain and boundary configuration

    Returns:
        List[Cell]: Reflex-tagged simulation grid
    """
    validate_config(config)

    domain = config["domain_definition"]
    nx, ny, nz, x_min, x_max, y_min, y_max, z_min, z_max = extract_domain_bounds(domain)

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    raw_grid = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = x_min + i * dx
                y = y_min + j * dy
                z = z_min + k * dz
                cell = Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=True)
                raw_grid.append(cell)

    if debug:
        print(f"[GRID] Constructed raw grid with {len(raw_grid)} cells")
        for idx, cell in enumerate(raw_grid):
            print(f"[GRID] Cell[{idx}] @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f})")

    grid = initialize_masks(raw_grid, config)

    if debug:
        fluid_count = sum(1 for c in grid if c.fluid_mask)
        ghost_count = len(grid) - fluid_count
        print(f"[GRID] Final grid → fluid={fluid_count}, ghost={ghost_count}")

    return grid
