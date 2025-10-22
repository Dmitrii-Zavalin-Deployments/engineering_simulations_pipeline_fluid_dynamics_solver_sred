# src/snapshot_manager.py
# 🧾 Snapshot Manager — orchestrates Navier-Stokes evolution and snapshot generation

import os
import logging
from src.step_controller import evolve_step
from src.utils.snapshot_step_processor import process_snapshot_step
from src.exporters.velocity_field_writer import write_velocity_field  # ✅ NEW

def generate_snapshots(input_data: dict, scenario_name: str, config: dict) -> list:
    """
    Executes the full Navier-Stokes simulation loop.

    Roadmap Alignment:
    Governing Equations:
        - Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u
        - Continuity: ∇ · u = 0

    Modular Execution:
        - Input parsing → input_reader.py
        - Grid generation → grid_generator.py
        - Boundary enforcement → boundary_condition_solver.py
        - Ghost logic → ghost_cell_generator.py, ghost_influence_applier.py
        - Momentum update → momentum_solver.py
            - Advection: u · ∇u → advection.py
            - Viscosity: μ∇²u → viscosity.py
        - Pressure solve → pressure_solver.py
        - Velocity projection → velocity_projection.py
        - Time loop orchestration → step_controller.py
        - Snapshot packaging → snapshot_step_processor.py

    Reflex Integration:
        - Reflex scoring injected per step
        - Mutation diagnostics tracked
        - Projection status logged
        - Snapshot metadata supports continuity audit and CI overlays

    Args:
        input_data (dict): Parsed simulation input
        scenario_name (str): Scenario identifier
        config (dict): Reflex and diagnostic configuration

    Returns:
        list: List of (step_index, snapshot_dict) tuples
    """
    # 🧩 Extract simulation parameters
    time_step = input_data["simulation_parameters"]["time_step"]
    total_time = input_data["simulation_parameters"]["total_time"]
    output_interval = input_data["simulation_parameters"].get("output_interval", 1)
    if output_interval <= 0:
        logging.warning(f"⚠️ output_interval was set to {output_interval}. Using fallback of 1.")
        output_interval = 1

    domain = input_data["domain_definition"]
    initial_conditions = input_data["initial_conditions"]
    geometry = input_data.get("geometry_definition")

    print(f"🧩 Domain resolution: {domain['nx']}×{domain['ny']}×{domain['nz']}")
    print(f"⚙️  Output interval: {output_interval}")

    # 🧱 Grid initialization with optional geometry mask
    if geometry:
        from src.grid_generator import generate_grid_with_mask
        grid = generate_grid_with_mask(domain, initial_conditions, geometry)
        mask_flat = geometry.get("geometry_mask_flat", [])
        fluid_code = geometry.get("mask_encoding", {}).get("fluid", 1)
        expected_size = mask_flat.count(fluid_code)
    else:
        from src.grid_generator import generate_grid
        grid = generate_grid(domain, initial_conditions)
        expected_size = domain["nx"] * domain["ny"] * domain["nz"]

    # 📐 Compute spatial discretization
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)
    print(f"[DEBUG] Grid spacing → dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

    # 🔁 Time integration loop
    num_steps = int(total_time / time_step)
    snapshots = []
    mutation_report = {
        "pressure_mutated": 0,
        "velocity_projected": 0,
        "projection_skipped": 0
    }

    output_folder = os.path.join("data", "testing-input-output", "navier_stokes_output")
    os.makedirs(output_folder, exist_ok=True)

    for step in range(num_steps + 1):
        # 🚀 Evolve fluid state using full Navier-Stokes logic
        grid, reflex = evolve_step(grid, input_data, step, config=config)

        # 💨 Export velocity field before pressure correction
        write_velocity_field(grid, step)  # ✅ NEW

        # 🧾 Package snapshot with diagnostics and metadata
        grid, snapshot = process_snapshot_step(
            step=step,
            grid=grid,
            reflex=reflex,
            spacing=spacing,
            config=config,
            expected_size=expected_size,
            output_folder=output_folder
        )

        # ✅ Track reflex score injection
        score = snapshot.get("reflex_score", 0.0)
        print(f"[VERIFY] Injected reflex_score: {score} ({type(score)})")

        # 📊 Update mutation summary counters
        if snapshot.get("pressure_mutated", False):
            mutation_report["pressure_mutated"] += 1
        if snapshot.get("velocity_projected", True):
            mutation_report["velocity_projected"] += 1
        if snapshot.get("projection_skipped", False):
            mutation_report["projection_skipped"] += 1

        # 📤 Export snapshot at output interval
        if step % output_interval == 0:
            snapshots.append((step, snapshot))

    # 📋 Final simulation summary
    print("🧾 Final Simulation Summary:")
    print(f"   Pressure mutated steps   → {mutation_report['pressure_mutated']}")
    print(f"   Velocity projected steps → {mutation_report['velocity_projected']}")
    print(f"   Projection skipped steps → {mutation_report['projection_skipped']}")

    return snapshots



