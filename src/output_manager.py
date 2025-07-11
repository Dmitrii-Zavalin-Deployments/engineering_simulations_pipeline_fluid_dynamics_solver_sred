# src/output_manager.py

from dataclasses import asdict
from typing import Any

EXPECTED_SNAPSHOT_KEYS = [
    "step_index",
    "grid",
    "max_velocity",
    "max_divergence",
    "global_cfl",
    "overflow_detected",
    "damping_enabled",
    "projection_passes"
]

def validate_snapshot(snapshot: dict) -> bool:
    """
    Checks if the snapshot meets required schema and contains expected keys.

    Args:
        snapshot (dict): Snapshot dictionary generated during simulation

    Returns:
        bool: True if all expected keys are present, False otherwise
    """
    return all(key in snapshot for key in EXPECTED_SNAPSHOT_KEYS)

def serialize_snapshot(snapshot: dict, grid: list[Any]) -> dict:
    """
    Converts Cell objects into JSON-compatible dictionaries and attaches them to the snapshot.

    Args:
        snapshot (dict): Snapshot metrics and metadata
        grid (list): List of Cell instances

    Returns:
        dict: Snapshot with grid converted to JSON-safe format
    """
    snapshot["grid"] = [asdict(cell) for cell in grid]
    return snapshot



