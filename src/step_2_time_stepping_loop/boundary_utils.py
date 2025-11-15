# src/step_2_time_stepping/boundary_utils.py

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
    """

    role = cell.get("boundary_role")
    if role is None:
        # No boundary → return unchanged
        return state

    # Find matching boundary condition by role
    bc_list = config.get("boundary_conditions", [])
    bc_match = next((bc for bc in bc_list if bc.get("role") == role), None)

    if not bc_match:
        # No matching boundary condition → return unchanged
        return state

    # Apply overrides based on "apply_to"
    new_state = {
        "pressure": state["pressure"],
        "velocity": {
            "vx": state["velocity"]["vx"],
            "vy": state["velocity"]["vy"],
            "vz": state["velocity"]["vz"],
        },
    }

    if "velocity" in bc_match.get("apply_to", []):
        vel = bc_match.get("velocity")
        if vel and len(vel) == 3:
            new_state["velocity"]["vx"] = vel[0]
            new_state["velocity"]["vy"] = vel[1]
            new_state["velocity"]["vz"] = vel[2]

    if "pressure" in bc_match.get("apply_to", []):
        pres = bc_match.get("pressure")
        if pres is not None:
            new_state["pressure"] = pres

    return new_state



