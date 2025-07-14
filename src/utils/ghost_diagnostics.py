# src/utils/ghost_diagnostics.py
# 📐 Utility to analyze ghost cell distribution by boundary face

from collections import defaultdict

def analyze_ghost_registry(ghost_registry: dict) -> dict:
    """
    Returns ghost cell stats by face and total.

    Args:
        ghost_registry (dict): Maps ghost cell id to metadata

    Returns:
        dict: Summary with per-face counts and totals
    """
    face_counts = defaultdict(int)
    for cell_id, meta in ghost_registry.items():
        face = meta.get("face")
        if face:
            face_counts[face] += 1

    return {
        "total": len(ghost_registry),
        "per_face": dict(face_counts)
    }

def log_ghost_summary(ghost_registry: dict):
    """
    Logs ghost cell distribution to console for quick inspection.
    """
    summary = analyze_ghost_registry(ghost_registry)
    print(f"🧱 Ghost Cells: {summary['total']} total")
    for face, count in summary["per_face"].items():
        print(f"   {face}: {count}")

def inject_diagnostics(snapshot: dict, ghost_registry: dict) -> dict:
    """
    Optionally attach ghost diagnostics to snapshot.

    Args:
        snapshot (dict): Existing snapshot dictionary
        ghost_registry (dict): Registry to analyze

    Returns:
        dict: Updated snapshot with ghost diagnostics
    """
    diagnostics = analyze_ghost_registry(ghost_registry)
    snapshot["ghost_diagnostics"] = diagnostics
    return snapshot



