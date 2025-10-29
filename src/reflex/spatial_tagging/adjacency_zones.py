# src/reflex/spatial_tagging/adjacency_zones.py
# 🧭 Adjacency Zones — tags fluid cells near ghost cells for reflex overlays and mutation proximity scoring
# 📌 This module extracts ghost coordinates for adjacency tagging and reflex diagnostics.
# It does NOT exclude based on adjacency or proximity — only ghost cell geometry is used.

from typing import Set, Tuple

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def extract_ghost_coordinates(ghost_registry) -> Set[Tuple[float, float, float]]:
    """
    Extracts ghost coordinates from registry.

    Args:
        ghost_registry (dict or set): Ghost metadata or ghost cell set

    Returns:
        Set[Tuple]: Coordinates of ghost cells
    """
    coords = set()
    if isinstance(ghost_registry, dict):
        for meta in ghost_registry.values():
            coord = meta.get("coordinate")
            if isinstance(coord, tuple):
                coords.add(coord)
                if debug:
                    print(f"[ADJACENCY] Extracted ghost coord from registry → {coord}")
    elif isinstance(ghost_registry, set):
        for cell in ghost_registry:
            coord = (cell.x, cell.y, cell.z)
            coords.add(coord)
            if debug:
                print(f"[ADJACENCY] Extracted ghost coord from cell → {coord}")
    return coords



