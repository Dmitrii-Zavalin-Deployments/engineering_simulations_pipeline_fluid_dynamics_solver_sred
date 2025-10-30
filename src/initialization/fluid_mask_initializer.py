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
        Tuple of (nx, ny, nz, min_x, max_x, min_y, max_y, min_z, max_z)
    """
    required_keys = [
        "nx", "ny", "nz",
        "min_x", "max_x",
        "min_y", "max_y",
        "min_z", "max_z"
    ]
    missing = [key for key in required_keys if key not in domain]
    if missing:
        raise ValueError(f"Missing required domain keys: {', '.join(missing)}")

    return (
        domain["nx"], domain["ny"], domain["nz"],
        domain["min_x"], domain["max_x"],
        domain["min_y"], domain["max_y"],
        domain["min_z"], domain["max_z"]
    )


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

    domain = config.get("domain_definition", {})
    ghost_rules = config.get("ghost_rules", {})
    boundary_faces = ghost_rules.get("boundary_faces", [])
    ghost_type_default = ghost_rules.get("default_type", "generic")

    _, _, _, min_x, max_x, min_y, max_y, min_z, max_z = extract_domain_bounds(domain)

    initialized = []

    for cell in grid:
        x, y, z = cell.x, cell.y, cell.z
        fluid = True
        ghost_face = None
        ghost_type = ghost_type_default
        boundary_tag = None

        # Detect ghost boundaries
        if x <= min_x:
            ghost_face = "xmin"
        elif x >= max_x:
            ghost_face = "xmax"
        elif y <= min_y:
            ghost_face = "ymin"
        elif y >= max_y:
            ghost_face = "ymax"
        elif z <= min_z:
            ghost_face = "zmin"
        elif z >= max_z:
            ghost_face = "zmax"

        if ghost_face:
            fluid = False
            boundary_tag = ghost_face

            # ✅ Normalize ghost_face for lookup compatibility
            normalized_face = ghost_face.replace("_", "").lower()
            ghost_type = ghost_rules.get("face_types", {}).get(normalized_face, ghost_type_default)

        initialized_cell = Cell(
            x=x,
            y=y,
            z=z,
            velocity=cell.velocity,
            pressure=cell.pressure,
            fluid_mask=fluid
        )

        if not fluid:
            initialized_cell.ghost_face = ghost_face
            initialized_cell.boundary_tag = boundary_tag
            initialized_cell.ghost_type = ghost_type
            initialized_cell.ghost_source_step = config.get("step_index", None)
            initialized_cell.was_enforced = ghost_face in boundary_faces
            initialized_cell.originated_from_boundary = True
            initialized_cell.mutation_triggered_by = "boundary_enforcement"  # ✅ Reflex traceability

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

    domain = config.get("domain_definition", {})
    nx, ny, nz, min_x, max_x, min_y, max_y, min_z, max_z = extract_domain_bounds(domain)

    dx = (max_x - min_x) / nx
    dy = (max_y - min_y) / ny
    dz = (max_z - min_z) / nz

    raw_grid = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = min_x + i * dx
                y = min_y + j * dy
                z = min_z + k * dz
                cell = Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
                raw_grid.append(cell)

    if debug:
        print(f"[GRID] Constructed raw grid with {len(raw_grid)} cells")

    # ✅ Apply reflex-aware fluid/ghost tagging
    grid = initialize_masks(raw_grid, config)

    if debug:
        fluid_count = sum(1 for c in grid if c.fluid_mask)
        ghost_count = len(grid) - fluid_count
        print(f"[GRID] Final grid → fluid={fluid_count}, ghost={ghost_count}")

    return grid
