# src/step_2_time_stepping_loop/boundary_utils.py

class BoundaryConditionError(Exception):
    """Custom exception for boundary condition validation errors."""
    pass


def enforce_boundary(state: dict, cell: dict, config: dict) -> dict:
    """
    Enforce boundary conditions for a single cell.

    Parameters
    ----------
    state : dict
        Current physics state of the cell:
        {
          "pressure": float,
          "velocity": {"vx": float, "vy": float, "vz": float}
        }
    cell : dict
        Cell dictionary entry from cell_dict, must include "boundary_role".
    config : dict
        Full simulation config, must include "boundary_conditions".

    Returns
    -------
    dict
        Updated physics state with boundary overrides applied.

    Raises
    ------
    BoundaryConditionError
        If required fields are missing in state or boundary condition.
    """

    # --- Validate state structure ---
    if "pressure" not in state:
        raise BoundaryConditionError("State validation failed: 'pressure' field is missing.")
    if "velocity" not in state:
        raise BoundaryConditionError("State validation failed: 'velocity' field is missing.")
    for comp in ("vx", "vy", "vz"):
        if comp not in state["velocity"]:
            raise BoundaryConditionError(f"State validation failed: velocity component '{comp}' is missing.")

    role = cell.get("boundary_role")
    if role is None:
        # No boundary → return unchanged
        return state

    # --- Validate boundary_conditions list ---
    bc_list = config.get("boundary_conditions")
    if not bc_list:
        raise BoundaryConditionError("Configuration validation failed: 'boundary_conditions' list is missing or empty.")

    # Find matching boundary condition by role
    bc_match = next((bc for bc in bc_list if bc.get("role") == role), None)
    if not bc_match:
        raise BoundaryConditionError(f"No boundary condition found for role '{role}'.")

    # --- Validate boundary condition fields ---
    if "apply_to" not in bc_match:
        raise BoundaryConditionError(f"Boundary condition for role '{role}' is missing 'apply_to' field.")
    if not isinstance(bc_match["apply_to"], list):
        raise BoundaryConditionError(f"Boundary condition for role '{role}' has invalid 'apply_to' type (must be list).")

    # --- Apply overrides ---
    new_state = {
        "pressure": state["pressure"],
        "velocity": {
            "vx": state["velocity"]["vx"],
            "vy": state["velocity"]["vy"],
            "vz": state["velocity"]["vz"],
        },
    }

    if "velocity" in bc_match["apply_to"]:
        vel = bc_match.get("velocity")
        if vel is None:
            raise BoundaryConditionError(f"Boundary condition for role '{role}' requires 'velocity' but it is missing.")
        if len(vel) != 3:
            raise BoundaryConditionError(f"Boundary condition for role '{role}' has invalid 'velocity' length (expected 3).")
        new_state["velocity"]["vx"], new_state["velocity"]["vy"], new_state["velocity"]["vz"] = vel

    if "pressure" in bc_match["apply_to"]:
        pres = bc_match.get("pressure")
        if pres is None:
            raise BoundaryConditionError(f"Boundary condition for role '{role}' requires 'pressure' but it is missing.")
        new_state["pressure"] = pres

    return new_state



