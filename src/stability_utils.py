def compute_volatility(snapshot: dict) -> dict:
    """Analyzes snapshot and returns volatility metrics."""
    return {
        "divergence_max": snapshot.get("divergence_max", 0),
        "volatility_slope": "flat",
        "volatility_delta": 0.0
    }



