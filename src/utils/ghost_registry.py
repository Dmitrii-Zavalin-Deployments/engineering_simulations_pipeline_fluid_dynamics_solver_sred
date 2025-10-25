# src/utils/ghost_registry.py
# 🧭 Ghost Registry — centralizes ghost cell metadata for reflex scoring, mutation tagging, and diagnostics
# 📌 This module builds a registry of ghost cells for adjacency scoring and reflex overlays.
# It includes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from typing import List, Dict, Tuple
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def build_ghost_registry(grid: List[Cell], verbose: bool = False) -> Dict[int, Dict]:
    """
    Constructs a registry of ghost cells with coordinate and tagging metadata.

    Roadmap Alignment:
    Reflex Diagnostics:
    - Centralizes ghost metadata for adjacency scoring and mutation traceability
    - Supports suppression zone detection and boundary enforcement

    Args:
        grid (List[Cell]): Full simulation grid
        verbose (bool): If True, prints debug info

    Returns:
        Dict[int, Dict]: Registry keyed by cell ID with coordinate and ghost flags
    """
    registry = {}

    for cell in grid:
        if not getattr(cell, "fluid_mask", True):
            entry = {
                "coordinate": (cell.x, cell.y, cell.z),
                "ghost_face": getattr(cell, "ghost_face", None),
                "boundary_tag": getattr(cell, "boundary_tag", None),
                "ghost_type": getattr(cell, "ghost_type", "generic"),
                "source_step": getattr(cell, "ghost_source_step", None),
                "was_enforced": getattr(cell, "was_enforced", False),
                "originated_from_boundary": getattr(cell, "originated_from_boundary", False),
                "velocity": getattr(cell, "velocity", None),
                "pressure": getattr(cell, "pressure", None)
            }
            registry[id(cell)] = entry

            if debug and verbose:
                print(f"[REGISTRY] Ghost cell @ {entry['coordinate']} → face={entry['ghost_face']}, type={entry['ghost_type']}")

    return registry

def extract_ghost_coordinates(registry: Dict[int, Dict]) -> List[Tuple[float, float, float]]:
    """
    Extracts ghost cell coordinates from registry.

    Args:
        registry (Dict[int, Dict]): Ghost registry

    Returns:
        List[Tuple]: List of (x, y, z) coordinates
    """
    return [entry["coordinate"] for entry in registry.values() if "coordinate" in entry]

# ✅ Diagnostic utility for threshold proximity tracing
def log_proximity_debug(fluid_coord: float, boundary_coord: float, spacing: float, face: str):
    """
    Logs proximity delta and threshold comparison for ghost creation.

    Args:
        fluid_coord (float): Fluid cell coordinate (x, y, or z)
        boundary_coord (float): Boundary face coordinate
        spacing (float): Grid spacing (dx, dy, dz)
        face (str): Boundary face label (e.g., 'x_min')
    """
    if debug:
        delta = abs(fluid_coord - boundary_coord)
        threshold = 0.5 * spacing
        proximity = abs(delta - threshold)
        print(f"[PROXIMITY] Face={face} → |{fluid_coord:.6f} - {boundary_coord:.6f}| = {delta:.6f} → Δ={proximity:.6f} vs threshold={threshold:.6f}")



