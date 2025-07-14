# src/utils/ghost_diagnostics.py
# 📐 Utility to analyze ghost cell distribution, pressure overrides, and enforcement stats — fluid adjacency aware

from collections import defaultdict

def analyze_ghost_registry(ghost_registry, grid=None) -> dict:
    """
    Returns ghost cell stats including per-face count, total, pressure enforcement, velocity rules,
    and optionally fluid cell adjacency counts if grid is provided.

    Args:
        ghost_registry (dict or set): Maps ghost cell id to metadata OR contains ghost cell objects
        grid (List[Cell], optional): Full simulation grid for adjacency tracking

    Returns:
        dict: Summary with per-face breakdown and enforcement stats
    """
    face_counts = defaultdict(int)
    pressure_overrides = 0
    no_slip_enforced = 0
    adjacent_fluid_cells = set()

    ghost_coords = set()

    # Normalize ghost cell coordinates
    if isinstance(ghost_registry, dict):
        for meta in ghost_registry.values():
            coord = meta.get("coordinate")
            if isinstance(coord, tuple):
                ghost_coords.add(coord)
            face = meta.get("face")
            if face:
                face_counts[face] += 1
            if isinstance(meta.get("pressure"), (int, float)):
                pressure_overrides += 1
            if meta.get("velocity") == [0.0, 0.0, 0.0]:
                no_slip_enforced += 1
        total = len(ghost_registry)

    elif isinstance(ghost_registry, set):
        for cell in ghost_registry:
            coord = (cell.x, cell.y, cell.z)
            ghost_coords.add(coord)
            face = getattr(cell, "ghost_face", None)
            if face:
                face_counts[face] += 1
            if isinstance(getattr(cell, "pressure", None), (int, float)):
                pressure_overrides += 1
            velocity = getattr(cell, "velocity", None)
            if velocity == [0.0, 0.0, 0.0]:
                no_slip_enforced += 1
        total = len(ghost_registry)

    else:
        raise TypeError("❌ ghost_registry must be dict or set")

    # 🧭 Fluid–ghost adjacency check (tolerant coordinate proximity)
    if grid:
        fluid_coords = {
            (cell.x, cell.y, cell.z) for cell in grid
            if getattr(cell, "fluid_mask", False)
        }

        domain = getattr(grid[0], "domain_metadata", None)
        if domain:
            dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / domain.get("nx", 1)
            dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / domain.get("ny", 1)
            dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / domain.get("nz", 1)
        else:
            # Fallback spacing
            dx, dy, dz = 1.0, 1.0, 1.0

        def coords_are_neighbors(a, b, tol=1e-6):
            return (
                abs(a[0] - b[0]) <= dx + tol and
                abs(a[1] - b[1]) <= dy + tol and
                abs(a[2] - b[2]) <= dz + tol
            )

        for f_coord in fluid_coords:
            for g_coord in ghost_coords:
                if coords_are_neighbors(f_coord, g_coord):
                    adjacent_fluid_cells.add(f_coord)

    return {
        "total": total,
        "per_face": dict(face_counts),
        "pressure_overrides": pressure_overrides,
        "no_slip_enforced": no_slip_enforced,
        "fluid_cells_adjacent_to_ghosts": len(adjacent_fluid_cells)
    }

def log_ghost_summary(ghost_registry, grid=None):
    """
    Logs ghost cell diagnostics to console for quick inspection, including fluid adjacency if grid is provided.
    """
    summary = analyze_ghost_registry(ghost_registry, grid)
    print(f"🧱 Ghost Cells: {summary['total']} total")
    for face, count in summary["per_face"].items():
        print(f"   {face}: {count}")
    print(f"📐 Ghost Pressure Overrides: {summary['pressure_overrides']}")
    print(f"🧊 No-slip Velocity Enforced: {summary['no_slip_enforced']}")
    if "fluid_cells_adjacent_to_ghosts" in summary:
        print(f"🧭 Fluid cells bordering ghosts: {summary['fluid_cells_adjacent_to_ghosts']}")

def inject_diagnostics(snapshot: dict, ghost_registry, grid=None) -> dict:
    """
    Optionally attach ghost diagnostics to snapshot, including fluid adjacency.

    Args:
        snapshot (dict): Existing snapshot dictionary
        ghost_registry (dict or set): Registry to analyze
        grid (List[Cell], optional): Simulation grid for adjacency computation

    Returns:
        dict: Updated snapshot with embedded ghost diagnostics
    """
    diagnostics = analyze_ghost_registry(ghost_registry, grid)
    snapshot["ghost_diagnostics"] = diagnostics
    return snapshot



