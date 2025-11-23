# src/step_2_time_stepping_loop/force_utils.py
# 🌪️ Force Loader — read external forces from input schema and attach to solver grid
#
# External forces are defined in the input JSON under "external_forces":
# {
#   "force_vector": [Fx, Fy, Fz],
#   "force_units": "N/m^3",
#   "force_comment": "Body force per unit volume (Fx,Fy,Fz)"
# }
#
# This module extracts those values and provides them to mac_update_velocity.py.

from typing import Dict, Any

debug = False  # toggle for verbose logging


def load_external_forces(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Load external forces from the input configuration.
    Returns a dict with keys 'Fx', 'Fy', 'Fz' in solver units.
    Raises KeyError if 'external_forces' block is missing.
    """
    if "external_forces" not in config:
        raise KeyError(
            "Missing 'external_forces' in input configuration. "
            "Expected block with 'force_vector', 'force_units', 'force_comment'."
        )

    forces = config["external_forces"]

    vec = forces.get("force_vector")
    if not vec or len(vec) != 3:
        raise ValueError(
            f"Invalid or missing 'force_vector' in external_forces: {vec}. "
            "Expected [Fx, Fy, Fz]."
        )

    Fx, Fy, Fz = vec
    out = {"Fx": Fx, "Fy": Fy, "Fz": Fz}

    if debug:
        print(f"[Force Loader] External forces loaded: {out} ({forces.get('force_units')})")

    return out



