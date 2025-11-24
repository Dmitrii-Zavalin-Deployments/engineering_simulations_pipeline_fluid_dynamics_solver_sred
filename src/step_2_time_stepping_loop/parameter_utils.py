# src/step_2_time_stepping_loop/parameter_utils.py
# 📦 Parameter Loader — extract simulation parameters, fluid properties, domain spacings, and external forces
#
# Provides a single entry point for solver modules to access dt, rho, mu, dx, dy, dz, Fx, Fy, Fz.
# Raises explicit KeyError or ValueError if required blocks or fields are missing/invalid.

from typing import Dict, Any

debug = False  # toggle for verbose logging


def load_solver_parameters(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract dt, rho, mu, grid spacings, and external forces from input config.

    Returns:
        dict: keys 'dt', 'rho', 'mu', 'dx', 'dy', 'dz', 'Fx', 'Fy', 'Fz'.

    Raises:
        KeyError: if required blocks or fields are missing.
        ValueError: if values are invalid (None, negative, or zero where not allowed).
    """
    # --- Required blocks ---
    for block in ["simulation_parameters", "fluid_properties", "domain_definition", "external_forces"]:
        if block not in config:
            raise KeyError(f"Missing '{block}' in input configuration.")

    sim = config["simulation_parameters"]
    fluid = config["fluid_properties"]
    domain = config["domain_definition"]
    forces = config["external_forces"]

    # --- Scalars ---
    dt = sim.get("time_step")
    if dt is None or dt <= 0:
        raise ValueError(f"Invalid or missing 'time_step': {dt}")

    rho = fluid.get("density")
    if rho is None or rho <= 0:
        raise ValueError(f"Invalid or missing 'density': {rho}")

    mu = fluid.get("viscosity")
    if mu is None or mu < 0:
        raise ValueError(f"Invalid or missing 'viscosity': {mu}")

    # --- Domain ---
    try:
        nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]
        x_min, x_max = domain["x_min"], domain["x_max"]
        y_min, y_max = domain["y_min"], domain["y_max"]
        z_min, z_max = domain["z_min"], domain["z_max"]
    except KeyError as e:
        raise KeyError(f"Missing domain field: {e}")

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid resolution: nx={nx}, ny={ny}, nz={nz}")

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError(f"Computed invalid grid spacings: dx={dx}, dy={dy}, dz={dz}")

    # --- Forces ---
    vec = forces.get("force_vector")
    if not vec or len(vec) != 3:
        raise ValueError(f"Invalid or missing 'force_vector' in external_forces: {vec}")
    Fx, Fy, Fz = vec

    out = {
        "dt": dt, "rho": rho, "mu": mu,
        "dx": dx, "dy": dy, "dz": dz,
        "Fx": Fx, "Fy": Fy, "Fz": Fz
    }

    if debug:
        print(f"[Parameter Loader] Solver parameters loaded: {out}")

    return out



