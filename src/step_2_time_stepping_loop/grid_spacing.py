# src/step_2_time_stepping_loop/grid_spacing.py
# 🧱 Step 2: Grid Spacing — Compute per-axis deltas

from typing import Tuple, Literal

debug = False  # toggle to True for verbose GitHub Action logs


def compute_grid_spacings(
    config: dict,
    mode: Literal["nx", "nx_minus_one"] = "nx"
) -> Tuple[float, float, float]:
    """
    Compute per-axis spatial deltas (dx, dy, dz) from domain_definition.

    Parameters
    ----------
    config : dict
        Simulation configuration dict containing "domain_definition" with:
        - x_min, x_max, y_min, y_max, z_min, z_max : float
        - nx, ny, nz : int
    mode : {"nx", "nx_minus_one"}
        - "nx": spacing = (x_max - x_min) / nx      [matches your requested formula]
        - "nx_minus_one": spacing = (x_max - x_min) / (nx - 1)
          Use this if your grid indices represent node counts (common for
          cell-centered or staggered grids where there are nx nodes across the span).

    Returns
    -------
    (dx, dy, dz) : tuple of float
        Per-axis spatial spacing.

    Raises
    ------
    ValueError
        If required fields are missing, invalid, or produce non-positive spacings.

    Notes
    -----
    - For face-centered (staggered/MAC) velocities, you may still use uniform dx, dy, dz
      from this function. If your grid uses non-uniform spacing, replace this with
      per-cell/per-face arrays.
    - If (x_max - x_min)/nx, (y_max - y_min)/ny, (z_max - z_min)/nz are different,
      this function returns distinct dx, dy, dz — that’s expected and supported.
    """
    dd = config.get("domain_definition")
    if not isinstance(dd, dict):
        raise ValueError("Configuration error: 'domain_definition' must be a dict.")

    required_keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz")
    missing = [k for k in required_keys if k not in dd]
    if missing:
        raise ValueError(f"Configuration error: domain_definition missing keys: {missing}")

    x_min = dd["x_min"]; x_max = dd["x_max"]
    y_min = dd["y_min"]; y_max = dd["y_max"]
    z_min = dd["z_min"]; z_max = dd["z_max"]
    nx = dd["nx"]; ny = dd["ny"]; nz = dd["nz"]

    if debug:
        print(f"🔍 Domain definition: x=({x_min},{x_max}), y=({y_min},{y_max}), z=({z_min},{z_max}), "
              f"nx={nx}, ny={ny}, nz={nz}, mode={mode}")

    # Basic type/validity checks
    for name, v in (("nx", nx), ("ny", ny), ("nz", nz)):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Configuration error: {name} must be a positive integer, got {v}.")

    for name, lo, hi in (("x", x_min, x_max), ("y", y_min, y_max), ("z", z_min, z_max)):
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
            raise ValueError(f"Configuration error: {name}_min/{name}_max must be numeric.")
        if hi <= lo:
            raise ValueError(f"Configuration error: {name}_max must be greater than {name}_min.")

    # Choose denominator based on mode
    denom_x = nx if mode == "nx" else (nx - 1)
    denom_y = ny if mode == "nx" else (ny - 1)
    denom_z = nz if mode == "nx" else (nz - 1)

    if denom_x <= 0 or denom_y <= 0 or denom_z <= 0:
        raise ValueError("Configuration error: nx_minus_one mode requires nx, ny, nz >= 2.")

    dx = (x_max - x_min) / float(denom_x)
    dy = (y_max - y_min) / float(denom_y)
    dz = (z_max - z_min) / float(denom_z)

    if debug:
        print(f"📐 Computed spacings: dx={dx}, dy={dy}, dz={dz}")

    # Final sanity checks
    for name, h in (("dx", dx), ("dy", dy), ("dz", dz)):
        if h <= 0.0 or not (h < float("inf")):
            raise ValueError(f"Configuration error: computed {name} must be > 0 and finite, got {h}.")

    if debug:
        print("✅ Grid spacing computation complete.")

    return dx, dy, dz



