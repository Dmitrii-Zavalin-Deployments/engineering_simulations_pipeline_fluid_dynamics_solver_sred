# src/step1_input_validation/config_validator.py

# ✅ Config Validator — ensures required fields for reflex-aware grid initialization are present and valid
# 📌 This module validates the structure of the simulation configuration.
# It does NOT interact with fluid_mask or geometry masking logic directly.
# It is NOT responsible for solver inclusion/exclusion decisions.

from typing import Dict

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def validate_config(config: Dict) -> None:
    """
    Validates the configuration object passed to initialize_masks().

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary.")

    # ✅ Validate domain_definition
    domain = config.get("domain_definition")
    if not isinstance(domain, dict):
        raise ValueError("Missing or invalid 'domain_definition' in config.")

    for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]:
        if key not in domain or not isinstance(domain[key], (int, float)):
            raise ValueError(f"Missing or invalid '{key}' in 'domain_definition'.")

    # ✅ Validate fluid_properties
    fluid = config.get("fluid_properties")
    if not isinstance(fluid, dict):
        raise ValueError("Missing or invalid 'fluid_properties' in config.")
    for key in ["density", "viscosity"]:
        if key not in fluid or not isinstance(fluid[key], (int, float)):
            raise ValueError(f"Missing or invalid '{key}' in 'fluid_properties'.")

    # ✅ Validate initial_conditions
    init = config.get("initial_conditions")
    if not isinstance(init, dict):
        raise ValueError("Missing or invalid 'initial_conditions' in config.")
    if "initial_velocity" not in init or not isinstance(init["initial_velocity"], list):
        raise ValueError("Missing or invalid 'initial_velocity' in 'initial_conditions'.")
    if "initial_pressure" not in init or not isinstance(init["initial_pressure"], (int, float)):
        raise ValueError("Missing or invalid 'initial_pressure' in 'initial_conditions'.")

    # ✅ Validate simulation_parameters
    sim = config.get("simulation_parameters")
    if not isinstance(sim, dict):
        raise ValueError("Missing or invalid 'simulation_parameters' in config.")
    for key in ["time_step", "total_time", "output_interval"]:
        if key not in sim or not isinstance(sim[key], (int, float)):
            raise ValueError(f"Missing or invalid '{key}' in 'simulation_parameters'.")

    # ✅ Validate boundary_conditions block
    boundary_conditions = config.get("boundary_conditions")
    if not isinstance(boundary_conditions, list):
        raise ValueError("Missing or invalid 'boundary_conditions' — must be a list.")

    for i, bc in enumerate(boundary_conditions):
        if not isinstance(bc, dict):
            raise ValueError(f"boundary_conditions[{i}] must be a dictionary.")
        for key in ["role", "type", "apply_to", "apply_faces"]:
            if key not in bc:
                raise ValueError(f"boundary_conditions[{i}] missing required key: '{key}'")
        if not isinstance(bc["apply_to"], list):
            raise ValueError(f"boundary_conditions[{i}] 'apply_to' must be a list.")
        if not isinstance(bc["type"], str):
            raise ValueError(f"boundary_conditions[{i}] 'type' must be a string.")
        if "velocity" in bc and not isinstance(bc["velocity"], list):
            raise ValueError(f"boundary_conditions[{i}] 'velocity' must be a list if present.")
        if "pressure" in bc and not isinstance(bc["pressure"], (int, float)):
            raise ValueError(f"boundary_conditions[{i}] 'pressure' must be numeric if present.")
        if "no_slip" in bc and not isinstance(bc["no_slip"], bool):
            raise ValueError(f"boundary_conditions[{i}] 'no_slip' must be boolean if present.")

    # ✅ Optional: ghost_rules
    ghost_rules = config.get("ghost_rules")
    if ghost_rules:
        if not isinstance(ghost_rules, dict):
            raise ValueError("Invalid 'ghost_rules' — must be a dictionary.")
        for key in ["boundary_faces", "default_type", "face_types"]:
            if key not in ghost_rules:
                raise ValueError(f"Missing '{key}' in 'ghost_rules'.")
        if not isinstance(ghost_rules["boundary_faces"], list):
            raise ValueError("Invalid 'boundary_faces' — must be a list.")
        if not isinstance(ghost_rules["default_type"], str):
            raise ValueError("Invalid 'default_type' — must be a string.")
        if not isinstance(ghost_rules["face_types"], dict):
            raise ValueError("Invalid 'face_types' — must be a dictionary.")

    if debug:
        print("[CONFIG] Validation passed — config is structurally complete.")
