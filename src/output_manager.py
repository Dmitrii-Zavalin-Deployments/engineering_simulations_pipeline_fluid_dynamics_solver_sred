# src/output_manager.py

from dataclasses import asdict
from typing import Any

def validate_snapshot(snapshot: dict) -> bool:
    """
    Checks if the snapshot meets required schema and contains expected fields.

    Args:
        snapshot (dict): Snapshot dictionary generated during simulation

    Returns:
        bool: True if all required keys are present, False otherwise
    """
    required_keys = [
        "divergence_max",
        "velocity_max",
        "overflow_flag",
        "reflex_triggered",
        "projection_passes",
        "volatility_slope",
        "volatility_delta",
        "damping_applied",
        "step_index",
        "timestamp",
        "grid"
    ]
    return all(key in snapshot for key in required_keys)


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



