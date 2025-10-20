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

    # ✅ Validate ghost_rules
    ghost_rules = config.get("ghost_rules")
    if not isinstance(ghost_rules, dict):
        raise ValueError("Missing or invalid 'ghost_rules' in config.")

    if "boundary_faces" not in ghost_rules or not isinstance(ghost_rules["boundary_faces"], list):
        raise ValueError("Missing or invalid 'boundary_faces' in 'ghost_rules'.")

    if "default_type" not in ghost_rules or not isinstance(ghost_rules["default_type"], str):
        raise ValueError("Missing or invalid 'default_type' in 'ghost_rules'.")

    if "face_types" not in ghost_rules or not isinstance(ghost_rules["face_types"], dict):
        raise ValueError("Missing or invalid 'face_types' in 'ghost_rules'.")

    print("[CONFIG] Validation passed — config is structurally complete.")



