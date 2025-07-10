# src/input_reader.py

import os
import json

def load_simulation_input(filepath: str) -> dict:
    """
    Reads and parses the fluid simulation input JSON file.
    Returns a structured dictionary with all validated fields.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Input file not found: {filepath}")

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Failed to parse JSON: {e}")

    required_sections = [
        "domain_definition",
        "fluid_properties",
        "initial_conditions",
        "simulation_parameters",
        "boundary_conditions"
    ]

    for section in required_sections:
        if section not in data:
            raise KeyError(f"❌ Missing required section: {section}")

    # Optional: Log scenario resolution and basic metadata
    domain = data["domain_definition"]
    nx, ny, nz = domain.get("nx"), domain.get("ny"), domain.get("nz")
    print(f"🧩 Domain resolution: {nx}×{ny}×{nz}")

    print(f"⚙️  Output interval: {data['simulation_parameters'].get('output_interval', 'N/A')}")

    return data



