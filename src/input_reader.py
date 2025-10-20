# src/input_reader.py
# 📥 Input Reader — parses and validates structured Navier-Stokes simulation input

import os
import json

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
    bc_list = data.get("boundary_conditions", [])
    for bc in bc_list:
        if isinstance(bc, dict):
            print(f"🚧 Boundary Conditions → Apply To: {bc.get('apply_to', [])}")
            print(f"   Velocity Enforced: {bc.get('velocity')}")
            print(f"   Pressure Enforced: {bc.get('pressure')}")
            print(f"   No-Slip Mode: {bc.get('no_slip', False)}")
        else:
            print(f"⚠️ Unexpected boundary condition format: {type(bc)} → {bc}")

    # 👻 Ghost Rule Logging — confirms ghost logic schema
    ghost_cfg = data.get("ghost_rules", {})
    print(f"👻 Ghost Rules → Faces: {ghost_cfg.get('boundary_faces', [])}")
    print(f"   Default Type: {ghost_cfg.get('default_type')}")
    print(f"   Face Types: {ghost_cfg.get('face_types', {})}")

    # 🧱 Geometry Masking (optional) — defines fluid vs solid topology
    geometry = data.get("geometry_definition")
    if geometry:
        shape = geometry.get("geometry_mask_shape")
        encoding = geometry.get("mask_encoding", {})
        print(f"🧱 Geometry mask shape: {shape}")
        print(f"🧱 Mask encoding: fluid={encoding.get('fluid')}, solid={encoding.get('solid')}")
        print(f"🧱 Flattening order: {geometry.get('flattening_order', 'x-major')}")

    return data



