# src/utils/ghost_diagnostics.py
# 📐 Utility to analyze ghost cell distribution, pressure overrides, and enforcement stats

from collections import defaultdict

def analyze_ghost_registry(ghost_registry) -> dict:
    """
    Returns ghost cell stats including per-face count, total, pressure enforcement, and velocity rules.

    Args:
        ghost_registry (dict or set): Maps ghost cell id to metadata OR contains ghost cell objects

    Returns:
        dict: Summary with per-face breakdown and enforcement stats
    """
    face_counts = defaultdict(int)
    pressure_overrides = 0
    no_slip_enforced = 0

    # If registry is a dict of id → metadata
    if isinstance(ghost_registry, dict):
        for cell_id, meta in ghost_registry.items():
            face = meta.get("face")
            if face:
                face_counts[face] += 1
            if isinstance(meta.get("pressure"), (int, float)):
                pressure_overrides += 1
            if meta.get("velocity") == [0.0, 0.0, 0.0]:
                no_slip_enforced += 1
        total = len(ghost_registry)

    # If registry is a set of ghost cell objects
    elif isinstance(ghost_registry, set):
        for cell in ghost_registry:
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

    return {
        "total": total,
        "per_face": dict(face_counts),
        "pressure_overrides": pressure_overrides,
        "no_slip_enforced": no_slip_enforced
    }

def log_ghost_summary(ghost_registry):
    """
    Logs ghost cell diagnostics to console for quick inspection.
    """
    summary = analyze_ghost_registry(ghost_registry)
    print(f"🧱 Ghost Cells: {summary['total']} total")
    for face, count in summary["per_face"].items():
        print(f"   {face}: {count}")
    print(f"📐 Ghost Pressure Overrides: {summary['pressure_overrides']}")
    print(f"🧊 No-slip Velocity Enforced: {summary['no_slip_enforced']}")

def inject_diagnostics(snapshot: dict, ghost_registry) -> dict:
    """
    Optionally attach ghost diagnostics to snapshot.

    Args:
        snapshot (dict): Existing snapshot dictionary
        ghost_registry (dict or set): Registry to analyze

    Returns:
        dict: Updated snapshot with embedded ghost diagnostics
    """
    diagnostics = analyze_ghost_registry(ghost_registry)
    snapshot["ghost_diagnostics"] = diagnostics
    return snapshot



