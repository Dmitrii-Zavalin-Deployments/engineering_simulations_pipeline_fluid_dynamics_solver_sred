# src/utils/ghost_diagnostics.py
# 📐 Utility to analyze ghost cell distribution, pressure overrides, and enforcement stats — fluid adjacency aware
# 📌 This module analyzes ghost-face enforcement and fluid adjacency for reflex overlays and diagnostics.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from collections import defaultdict

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def analyze_ghost_registry(ghost_registry, grid=None, spacing=(1.0, 1.0, 1.0)) -> dict:
    """
    Returns ghost cell stats including per-face count, total, pressure enforcement, velocity rules,
    and optionally fluid cell adjacency counts if grid is provided.
    """
    face_counts = defaultdict(int)
    pressure_overrides = 0
    no_slip_enforced = 0
    adjacent_fluid_cells = set()
    ghost_coords = set()

    # Normalize ghost cell coordinates and collect metadata
    if isinstance(ghost_registry, dict):
        for meta in ghost_registry.values():
            coord = meta.get("coordinate")
            if isinstance(coord, tuple):
                ghost_coords.add(coord)
            face = meta.get("face") or meta.get("ghost_face")
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

    # 🧭 Fluid–ghost adjacency detection using tolerant physical proximity
    if grid:
        dx, dy, dz = spacing
        if debug:
            print(f"[GHOST] Spacing used → dx={dx}, dy={dy}, dz={dz}")
            print(f"[GHOST] Total ghost cells: {len(ghost_coords)}")

        def coords_are_neighbors(a, b, tol=1e-3):
            return (
                abs(a[0] - b[0]) <= dx + tol and
                abs(a[1] - b[1]) <= dy + tol and
                abs(a[2] - b[2]) <= dz + tol
            )

        for cell in grid:
            if not getattr(cell, "fluid_mask", False):
                continue
            fluid_coord = (cell.x, cell.y, cell.z)
            for g_coord in ghost_coords:
                if coords_are_neighbors(fluid_coord, g_coord):
                    if debug:
                        print(f"[GHOST] Fluid {fluid_coord} ↔ Ghost {g_coord} → adjacent")
                    adjacent_fluid_cells.add(fluid_coord)

                    # ✅ Patch: context-based influence tagging
                    ghost_meta = None
                    if isinstance(ghost_registry, dict):
                        ghost_meta = next((meta for meta in ghost_registry.values() if meta.get("coordinate") == g_coord), None)
                    elif isinstance(ghost_registry, set):
                        ghost_meta = next((g for g in ghost_registry if (g.x, g.y, g.z) == g_coord), None)

                    if ghost_meta:
                        was_enforced = getattr(ghost_meta, "was_enforced", False)
                        from_boundary = getattr(ghost_meta, "originated_from_boundary", False)
                        if was_enforced or from_boundary:
                            cell.influenced_by_ghost = True
                            cell.mutation_triggered_by = "ghost_influence"

    return {
        "total": total,
        "per_face": dict(face_counts),
        "pressure_overrides": pressure_overrides,
        "no_slip_enforced": no_slip_enforced,
        "fluid_cells_adjacent_to_ghosts": len(adjacent_fluid_cells)
    }

def log_ghost_summary(ghost_registry, grid=None, spacing=(1.0, 1.0, 1.0)):
    """
    Logs ghost cell diagnostics to console for quick inspection, including fluid adjacency if grid is provided.
    """
    summary = analyze_ghost_registry(ghost_registry, grid, spacing)
    if debug:
        print(f"🧱 Ghost Cells: {summary['total']} total")
        for face, count in summary["per_face"].items():
            print(f"   {face}: {count}")
        print(f"📐 Ghost Pressure Overrides: {summary['pressure_overrides']}")
        print(f"🧊 No-slip Velocity Enforced: {summary['no_slip_enforced']}")
        print(f"🧭 Fluid cells bordering ghosts: {summary['fluid_cells_adjacent_to_ghosts']}")

def inject_diagnostics(snapshot: dict, ghost_registry, grid=None, spacing=(1.0, 1.0, 1.0)) -> dict:
    """
    Optionally attach ghost diagnostics to snapshot, including fluid adjacency.
    Logs diagnostics immediately after injecting.
    """
    diagnostics = analyze_ghost_registry(ghost_registry, grid, spacing)
    snapshot["ghost_diagnostics"] = diagnostics
    snapshot["ghost_registry"] = ghost_registry  # ✅ Embed registry for downstream traceability
    log_ghost_summary(ghost_registry, grid, spacing)
    return snapshot



