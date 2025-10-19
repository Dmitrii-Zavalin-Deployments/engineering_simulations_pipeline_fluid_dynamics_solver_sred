# src/input_reader.py
# 📥 Input Reader — parses and validates structured Navier-Stokes simulation input

import os
import json

def load_simulation_input(filepath: str) -> dict:
    """
    Parses the structured JSON input file for a Navier-Stokes simulation.

    Roadmap Alignment:
    1. Domain Definition → spatial bounds and grid resolution for discretization
    2. Fluid Properties → density (ρ) and viscosity (μ) for momentum equation
    3. Initial Conditions → velocity and pressure fields for ∂u/∂t initialization
    4. Simulation Parameters → time step, total time, output interval
    5. Boundary Conditions → enforcement logic for ghost cells and ∇P coupling
    6. Geometry Definition (optional) → fluid vs solid masking for domain topology

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
            raise KeyError(f"❌ Missing required section: {section}")

    # 🧩 Domain Definition — supports spatial discretization
    domain = data["domain_definition"]
    nx, ny, nz = domain.get("nx"), domain.get("ny"), domain.get("nz")
    bounds = (
        domain.get("min_x"), domain.get("max_x"),
        domain.get("min_y"), domain.get("max_y"),
        domain.get("min_z"), domain.get("max_z")
    )
    print(f"🧩 Domain resolution: {nx}×{ny}×{nz}")
    print(f"📐 Domain bounds: x={bounds[0]}→{bounds[1]}, y={bounds[2]}→{bounds[3]}, z={bounds[4]}→{bounds[5]}")

    # 🌊 Fluid Properties — used in momentum equation: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u
    fluid = data["fluid_properties"]
    print(f"🌊 Fluid density (ρ): {fluid.get('density', 'N/A')}")
    print(f"🌊 Fluid viscosity (μ): {fluid.get('viscosity', 'N/A')}")

    # 🌀 Initial Conditions — sets ∂u/∂t and pressure field at t=0
    init = data["initial_conditions"]
    print(f"🌀 Initial velocity: {init.get('initial_velocity', 'N/A')}")
    print(f"🌀 Initial pressure: {init.get('initial_pressure', 'N/A')}")

    # ⏱️ Simulation Parameters — controls time loop and output cadence
    sim = data["simulation_parameters"]
    print(f"⏱️ Time step (Δt): {sim.get('time_step', 'N/A')}")
    print(f"⏱️ Total time (T): {sim.get('total_time', 'N/A')}")
    print(f"⚙️ Output interval: {sim.get('output_interval', 'N/A')}")

    # 💧 Pressure Solver Configuration — governs ∇²P = ∇ · u enforcement
    pressure_cfg = data.get("pressure_solver", {})
    method = pressure_cfg.get("method", "jacobi")
    tolerance = pressure_cfg.get("tolerance", 1e-6)
    print(f"💧 Pressure Solver → Method: {method}, Tolerance: {tolerance}")

    # 🚧 Boundary Conditions — governs ghost logic and ∇P coupling
    bc = data["boundary_conditions"]
    print(f"🚧 Boundary Conditions → Apply To: {bc.get('apply_to', [])}")
    print(f"   Velocity Enforced: {bc.get('velocity')}")
    print(f"   Pressure Enforced: {bc.get('pressure')}")
    print(f"   No-Slip Mode: {bc.get('no_slip', False)}")

    # 🧱 Geometry Masking (optional) — defines fluid vs solid topology
    geometry = data.get("geometry_definition")
    if geometry:
        shape = geometry.get("geometry_mask_shape")
        encoding = geometry.get("mask_encoding", {})
        print(f"🧱 Geometry mask shape: {shape}")
        print(f"🧱 Mask encoding: fluid={encoding.get('fluid')}, solid={encoding.get('solid')}")
        print(f"🧱 Flattening order: {geometry.get('flattening_order', 'x-major')}")

    return data



