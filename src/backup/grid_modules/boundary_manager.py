# src/grid_modules/boundary_manager.py
# 🧱 Boundary Manager — tags edge cells based on grid resolution for diagnostic
# overlays and ghost logic
# 📌 This module identifies outermost grid faces using index-based logic.
# It does NOT exclude cells from solver routines.
# It does NOT interact with fluid_mask or geometry masking logic.

from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = False


def apply_boundaries(cells: list[Cell], domain: dict) -> list[Cell]:
    """
    Applies simplified boundary conditions by tagging edge cells using resolution values.
    Identifies cells on the outermost x/y/z faces based on grid indices.
    Does not require physical min/max domain coordinates.

    Args:
        cells (list[Cell]): Structured grid of cells
        domain (dict): Must contain "nx", "ny", "nz"

    Returns:
        list[Cell]: Original cell list (boundary tagging now handled via coordinate logic)
    """
    nx = domain.get("nx", 0)
    ny = domain.get("ny", 0)
    nz = domain.get("nz", 0)

    x_faces = {0, nx - 1} if nx > 0 else set()
    y_faces = {0, ny - 1} if ny > 0 else set()
    z_faces = {0, nz - 1} if nz > 0 else set()

    # ✅ Removed unused is_boundary assignment
    # Boundary tagging now handled via coordinate-based diagnostics in scoring
    # and overlays

    if debug:
        print(
            f"[BOUNDARY] Domain resolution → nx={nx}, ny={ny}, nz={nz}"
        )
        print(
            f"[BOUNDARY] Tagged boundary faces → x={x_faces}, y={y_faces}, "
            f"z={z_faces}"
        )

    return cells
