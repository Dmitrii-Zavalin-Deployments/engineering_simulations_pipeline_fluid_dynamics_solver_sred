# src/config/config_loader.py
# 🧠 Config Loader — centralizes domain + ghost rule injection for simulation setup
# 📌 Promotes modularity, reuse, and reflex-safe configuration handling

import json
import os

def load_simulation_config(domain_path: str, ghost_path: str, step_index: int = 0) -> dict:
    """
    Loads simulation configuration from domain and ghost rule JSON files.

    Args:
        domain_path (str): Path to domain definition JSON
        ghost_path (str): Path to ghost rule JSON
        step_index (int): Optional step index for reflex tagging

    Returns:
        dict: Combined configuration dictionary with domain, ghost_rules, and step_index
    """
    if not os.path.isfile(domain_path):
        raise FileNotFoundError(f"Domain config not found: {domain_path}")
    if not os.path.isfile(ghost_path):
        raise FileNotFoundError(f"Ghost rules config not found: {ghost_path}")

    # ✅ Unwrap domain_definition if nested
    with open(domain_path, "r") as f1:
        domain_raw = json.load(f1)
        domain = domain_raw.get("domain_definition", domain_raw)

    with open(ghost_path, "r") as f2:
        ghost_rules = json.load(f2)

    # ✅ Normalize ghost face keys if present
    face_types = ghost_rules.get("face_types", {})
    ghost_rules["face_types"] = {
        k.replace("_", "").lower(): v for k, v in face_types.items()
    }

    return {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "step_index": step_index
    }
