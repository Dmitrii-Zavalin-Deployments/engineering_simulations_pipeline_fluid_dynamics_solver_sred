# src/config/config_validator.py
# ✅ Config Validator — ensures required fields for reflex-aware grid initialization are present and valid

from typing import Dict

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

    for key in ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]:
        if key not in domain or not isinstance(domain[key], (int, float)):
            raise ValueError(f"Missing or invalid '{key}' in 'domain_definition'.")

    # ✅ Validate ghost_rules (optional if injected externally)
    ghost_rules = config.get("ghost_rules")
    if ghost_rules:
        if not isinstance(ghost_rules, dict):
            raise ValueError("Invalid 'ghost_rules' — must be a dictionary.")
        if "boundary_faces" not in ghost_rules or not isinstance(ghost_rules["boundary_faces"], list):
            raise ValueError("Missing or invalid 'boundary_faces' in 'ghost_rules'.")
        if "default_type" not in ghost_rules or not isinstance(ghost_rules["default_type"], str):
            raise ValueError("Missing or invalid 'default_type' in 'ghost_rules'.")
        if "face_types" not in ghost_rules or not isinstance(ghost_rules["face_types"], dict):
            raise ValueError("Missing or invalid 'face_types' in 'ghost_rules'.")

    # ✅ Validate boundary_conditions block
    boundary_conditions = config.get("boundary_conditions", [])
    if not isinstance(boundary_conditions, list):
        raise ValueError("Missing or invalid 'boundary_conditions' — must be a list.")

    for i, bc in enumerate(boundary_conditions):
        if not isinstance(bc, dict):
            raise ValueError(f"boundary_conditions[{i}] must be a dictionary.")
        if "apply_to" not in bc or not isinstance(bc["apply_to"], list):
            raise ValueError(f"boundary_conditions[{i}] missing or invalid 'apply_to' list.")
        if "type" not in bc or not isinstance(bc["type"], str):
            raise ValueError(f"boundary_conditions[{i}] missing or invalid 'type' string.")
        # Optional fields: velocity, pressure, no_slip
        if "velocity" in bc and not isinstance(bc["velocity"], list):
            raise ValueError(f"boundary_conditions[{i}] 'velocity' must be a list if present.")
        if "pressure" in bc and not isinstance(bc["pressure"], (int, float)):
            raise ValueError(f"boundary_conditions[{i}] 'pressure' must be numeric if present.")
        if "no_slip" in bc and not isinstance(bc["no_slip"], bool):
            raise ValueError(f"boundary_conditions[{i}] 'no_slip' must be boolean if present.")

    print("[CONFIG] Validation passed — config is structurally complete.")



