# src/input_reader.py

import os
import json

def load_simulation_input(filepath: str) -> dict:
    """
    Reads and parses the fluid simulation input JSON file.
    Returns a structured dictionary with all validated fields.
    Also logs key solver and boundary configuration metadata.
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

    # ✅ Domain resolution
    domain = data["domain_definition"]
    nx, ny, nz = domain.get("nx"), domain.get("ny"), domain.get("nz")
    print(f"🧩 Domain resolution: {nx}×{ny}×{nz}")

    # ✅ Output interval
    sim_params = data["simulation_parameters"]
    print(f"⚙️  Output interval: {sim_params.get('output_interval', 'N/A')}")

    # ✅ Pressure solver config
    pressure_cfg = data.get("pressure_solver", {})
    method = pressure_cfg.get("method", "jacobi")
    tolerance = pressure_cfg.get("tolerance", 1e-6)
    print(f"💧 Pressure Solver → Method: {method}, Tolerance: {tolerance}")

    # ✅ Boundary condition config
    bc = data["boundary_conditions"]
    print(f"🚧 Boundary Conditions → Apply To: {bc.get('apply_to', [])}")
    print(f"   Velocity Enforced: {bc.get('velocity')}")
    print(f"   Pressure Enforced: {bc.get('pressure')}")
    print(f"   No-Slip Mode: {bc.get('no_slip', False)}")

    return data



