# src/config/config_loader.py
# 🧠 Config Loader — centralizes domain + ghost rule injection for simulation setup
# 📌 Promotes modularity, reuse, and reflex-safe configuration handling

import json
import os


def load_simulation_config(
    domain_path: str,
    ghost_path: str,
    step_index: int = 0
) -> dict:
    """
    Loads simulation configuration from domain and ghost rule JSON files.

    Args:
        domain_path (str): Path to full simulation input JSON
        ghost_path (str): Path to ghost rule JSON
        step_index (int): Optional step index for reflex tagging

    Returns:
        dict: Combined configuration dictionary with ghost_rules and step_index injected
    """
    if not os.path.isfile(domain_path):
        raise FileNotFoundError(f"Domain config not found: {domain_path}")
    if not os.path.isfile(ghost_path):
        raise FileNotFoundError(f"Ghost rules config not found: {ghost_path}")

    # ✅ Load full simulation input (not just domain block)
    with open(domain_path, "r") as f1:
        input_data = json.load(f1)

    # ✅ Load and normalize ghost rules
    with open(ghost_path, "r") as f2:
        ghost_rules = json.load(f2)
    ghost_rules["face_types"] = {
        k.replace("_", "").lower(): v
        for k, v in ghost_rules.get("face_types", {}).items()
    }

    # ✅ Inject ghost rules and step index into full config
    input_data["ghost_rules"] = ghost_rules
    input_data["step_index"] = step_index

    return input_data
