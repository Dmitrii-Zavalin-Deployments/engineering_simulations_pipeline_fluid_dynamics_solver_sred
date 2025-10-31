# src/step1_input_validation/input_reader.py
# 📥 Input Reader — parses and validates structured Navier-Stokes simulation input
# 📌 This module anchors schema alignment for reflex scoring, mutation overlays, and diagnostic traceability.

import os
import json

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def load_simulation_input(filepath: str) -> dict:
    """
    Parses the structured JSON input file for a Navier-Stokes simulation.

    Roadmap Alignment:
    Schema → Solver Modules:
    1. Domain Definition → grid_generator.py, velocity_projection.py
    2. Fluid Properties → momentum_solver.py, viscosity.py
    3. Initial Conditions → grid_generator.py, advection.py
    4. Simulation Parameters → snapshot_manager.py, step_controller.py
    5. Boundary Conditions → boundary_condition_solver.py, ghost_influence_applier.py
    6. Pressure Solver → pressure_projection.py, jacobi.py
    7. Geometry Definition (optional) → grid_generator.py, ghost_cell_generator.py

    Reflex Integration:
    - Input parsing anchors diagnostic traceability
    - Schema alignment supports reflex scoring and mutation overlays

    Returns:
        dict: Validated input configuration
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Input file not found: {filepath}")

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Failed to parse JSON: {e}")

    # ✅ Required schema sections
    required_sections = [
        "domain_definition",
        "fluid_properties",
        "initial_conditions",
        "simulation_parameters",
        "boundary_conditions"
    ]
    for section in required_sections:
        if section not in data:
            raise KeyError(f"❌ Missing required section: '{section}'")

    # ✅ Domain definition
    domain = data["domain_definition"]
    for key in ["nx", "ny", "nz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]:
        if key not in domain:
            raise KeyError(f"❌ Missing domain key: '{key}'")
    if debug:
        print(f"🧩 Domain resolution: {domain['nx']}×{domain['ny']}×{domain['nz']}")
        print(f"📐 Domain bounds: x={domain['x_min']}→{domain['x_max']}, y={domain['y_min']}→{domain['y_max']}, z={domain['z_min']}→{domain['z_max']}")

    # ✅ Fluid properties
    fluid = data["fluid_properties"]
    for key in ["density", "viscosity"]:
        if key not in fluid:
            raise KeyError(f"❌ Missing fluid property: '{key}'")
    if debug:
        print(f"🌊 Fluid density (ρ): {fluid['density']}")
        print(f"🌊 Fluid viscosity (μ): {fluid['viscosity']}")

    # ✅ Initial conditions
    init = data["initial_conditions"]
    for key in ["initial_velocity", "initial_pressure"]:
        if key not in init:
            raise KeyError(f"❌ Missing initial condition: '{key}'")
    if debug:
        print(f"🌀 Initial velocity: {init['initial_velocity']}")
        print(f"🌀 Initial pressure: {init['initial_pressure']}")

    # ✅ Simulation parameters
    sim = data["simulation_parameters"]
    for key in ["time_step", "total_time", "output_interval"]:
        if key not in sim:
            raise KeyError(f"❌ Missing simulation parameter: '{key}'")
    if debug:
        print(f"⏱️ Time step (Δt): {sim['time_step']}")
        print(f"⏱️ Total time (T): {sim['total_time']}")
        print(f"⚙️ Output interval: {sim['output_interval']}")

    # ✅ Optional: Pressure solver config
    if "pressure_solver" in data:
        pressure_cfg = data["pressure_solver"]
        for key in ["method", "tolerance"]:
            if key not in pressure_cfg:
                raise KeyError(f"❌ Missing pressure solver key: '{key}'")
        if debug:
            print(f"💧 Pressure Solver → Method: {pressure_cfg['method']}, Tolerance: {pressure_cfg['tolerance']}")

    # ✅ Boundary conditions
    bc_list = data["boundary_conditions"]
    if not isinstance(bc_list, list):
        raise TypeError("❌ 'boundary_conditions' must be a list.")
    for i, bc in enumerate(bc_list):
        if not isinstance(bc, dict):
            raise TypeError(f"❌ boundary_conditions[{i}] must be a dictionary.")
        for key in ["role", "type", "apply_to", "apply_faces"]:
            if key not in bc:
                raise KeyError(f"❌ boundary_conditions[{i}] missing required key: '{key}'")
        if debug:
            print(f"🚧 Boundary Role: {bc['role']}")
            print(f"   Type: {bc['type']}")
            print(f"   Apply To: {bc['apply_to']}")
            print(f"   Apply Faces: {bc['apply_faces']}")
            print(f"   Velocity: {bc['velocity']}" if "velocity" in bc else "   Velocity: —")
            print(f"   Pressure: {bc['pressure']}" if "pressure" in bc else "   Pressure: —")
            print(f"   No-Slip: {bc['no_slip']}" if "no_slip" in bc else "   No-Slip: —")

    # ✅ Optional: Geometry definition
    if "geometry_definition" in data:
        geometry = data["geometry_definition"]
        for key in ["geometry_mask_flat", "geometry_mask_shape", "mask_encoding", "flattening_order"]:
            if key not in geometry:
                raise KeyError(f"❌ Missing geometry key: '{key}'")
        if debug:
            print(f"🧱 Geometry mask shape: {geometry['geometry_mask_shape']}")
            print(f"🧱 Mask encoding: fluid={geometry['mask_encoding']['fluid']}, solid={geometry['mask_encoding']['solid']}")
            print(f"🧱 Flattening order: {geometry['flattening_order']}")
            print(f"🧱 Mask preview (first 32): {geometry['geometry_mask_flat'][:32]}")

    return data
