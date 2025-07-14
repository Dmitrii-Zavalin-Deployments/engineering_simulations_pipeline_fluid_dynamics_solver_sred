# src/utils/ghost_diagnostics.py
# 📐 Utility to analyze ghost cell distribution by boundary face

from collections import defaultdict

def analyze_ghost_registry(ghost_registry) -> dict:
    """
    Returns ghost cell stats by face and total.

    Args:
        ghost_registry (dict or set): Maps ghost cell id to metadata OR contains ghost cells

    Returns:
        dict: Summary with per-face counts and totals
    """
    face_counts = defaultdict(int)

    # If registry is a dict of id → metadata
    if isinstance(ghost_registry, dict):
        for _, meta in ghost_registry.items():
            face = meta.get("face")
            if face:
                face_counts[face] += 1
        total = len(ghost_registry)

    # If registry is a set of ghost cells with .ghost_face attributes
    elif isinstance(ghost_registry, set):
        for cell in ghost_registry:
            face = getattr(cell, "ghost_face", None)
            if face:
                face_counts[face] += 1
        total = len(ghost_registry)

    else:
        raise TypeError("❌ ghost_registry must be dict or set")

    return {
        "total": total,
        "per_face": dict(face_counts)
    }

def log_ghost_summary(ghost_registry):
    """
    Logs ghost cell distribution to console for quick inspection.
    """
    summary = analyze_ghost_registry(ghost_registry)
    print(f"🧱 Ghost Cells: {summary['total']} total")
    for face, count in summary["per_face"].items():
        print(f"   {face}: {count}")

def inject_diagnostics(snapshot: dict, ghost_registry) -> dict:
    """
    Optionally attach ghost diagnostics to snapshot.

    Args:
        snapshot (dict): Existing snapshot dictionary
        ghost_registry (dict or set): Registry to analyze

    Returns:
        dict: Updated snapshot with ghost diagnostics
    """
    diagnostics = analyze_ghost_registry(ghost_registry)
    snapshot["ghost_diagnostics"] = diagnostics
    return snapshot



