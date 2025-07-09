def validate_snapshot(snapshot: dict) -> bool:
    """Checks if snapshot meets schema and content expectations."""
    required_keys = [
        "divergence_max", "velocity_max", "overflow_flag",
        "reflex_triggered", "projection_passes", "volatility_slope",
        "volatility_delta", "damping_applied", "step_index", "timestamp"
    ]
    return all(key in snapshot for key in required_keys)



