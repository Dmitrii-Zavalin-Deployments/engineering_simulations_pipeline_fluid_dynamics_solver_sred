# src/snapshot_manager.py
# 🧾 Snapshot Manager — orchestrates Navier-Stokes evolution and snapshot generation
# 📌 This module coordinates time integration, reflex scoring, and metadata packaging.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import os
import tempfile
from src.step_controller import evolve_step
from src.utils.snapshot_step_processor import process_snapshot_step
from src.exporters.velocity_field_writer import write_velocity_field

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def log_mutation_summary(mutation_report: dict):
    if not debug:
        return
    print("🧾 Final Simulation Summary:")
    print(f"   Pressure mutated steps   → {mutation_report['pressure_mutated']}")
    print(f"   Velocity projected steps → {mutation_report['velocity_projected']}")
    print(f"   Projection skipped steps → {mutation_report['projection_skipped']}")

def generate_snapshots(
    sim_config: dict,
    scenario_name: str,
    reflex_config: dict,
    output_dir: str | None = None
) -> list:
    """
    Executes the full Navier-Stokes simulation loop.
    """
    input_data = sim_config
    config = reflex_config

    sim_params = input_data.get("simulation_parameters")
    if sim_params is None:
        raise ValueError("Missing required block: 'simulation_parameters'")

    required_keys = ["time_step", "total_time", "output_interval"]
    missing_keys = [key for key in required_keys if key not in sim_params]
    if missing_keys:
        raise ValueError(f"Missing required simulation parameters: {', '.join(missing_keys)}")

    time_step = sim_params["time_step"]
    total_time = sim_params["total_time"]
    output_interval = sim_params["output_interval"]
    if output_interval <= 0:
        raise ValueError(f"Invalid output_interval: {output_interval}. Must be a positive integer.")

    domain = input_data.get("domain_definition")
    if domain is None:
        raise ValueError("Missing required block: 'domain_definition'")

    initial_conditions = input_data.get("initial_conditions")
    if initial_conditions is None:
        raise ValueError("Missing required block: 'initial_conditions'")

    geometry = input_data.get("geometry_definition")

    if debug:
        print(f"🧩 Domain resolution: {domain['nx']}×{domain['ny']}×{domain['nz']}")
        print(f"⚙️ Output interval: {output_interval}")

    if geometry:
        from src.grid_generator import generate_grid_with_mask
        grid = generate_grid_with_mask(domain, initial_conditions, geometry)
        mask_flat = geometry.get("geometry_mask_flat", [])
        fluid_code = geometry.get("mask_encoding", {}).get("fluid")
        if fluid_code is None:
            raise ValueError("Missing required mask_encoding key: 'fluid'")
        expected_size = mask_flat.count(fluid_code)
    else:
        from src.grid_generator import generate_grid
        grid = generate_grid(domain, initial_conditions)
        expected_size = domain["nx"] * domain["ny"] * domain["nz"]

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)
    if debug:
        print(f"[DEBUG] Grid spacing → dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

    num_steps = int(total_time / time_step)
    snapshots = []
    mutation_report = {
        "pressure_mutated": 0,
        "velocity_projected": 0,
        "projection_skipped": 0
    }

    output_folder = output_dir or tempfile.mkdtemp(prefix="navier_stokes_output_")
    os.makedirs(output_folder, exist_ok=True)

    for step in range(num_steps + 1):
        grid, reflex = evolve_step(grid, input_data, step, config=config, sim_config=sim_config)

        if debug:
            fluid_count = sum(1 for c in grid if getattr(c, "fluid_mask", False))
            print(f"[DEBUG] Velocity export: {fluid_count} fluid cells at step {step}")
        write_velocity_field(grid, step, output_dir=output_folder)
        if debug:
            print(f"[DEBUG] Velocity field written to: {os.path.join(output_folder, f'velocity_field_step_{step:04d}.json')}")

        grid, snapshot = process_snapshot_step(
            step=step,
            grid=grid,
            reflex=reflex,
            spacing=spacing,
            config=config,
            expected_size=expected_size,
            output_folder=output_folder,
            sim_config=sim_config
        )

        required_flags = ["reflex_score", "pressure_mutated", "velocity_projected", "projection_skipped"]
        missing_flags = [key for key in required_flags if key not in snapshot]
        if missing_flags:
            raise ValueError(f"Missing required snapshot flags: {', '.join(missing_flags)}")

        score = snapshot["reflex_score"]
        if debug:
            print(f"[VERIFY] Injected reflex_score: {score} ({type(score)})")

        if snapshot["pressure_mutated"]:
            mutation_report["pressure_mutated"] += 1
        if snapshot["velocity_projected"]:
            mutation_report["velocity_projected"] += 1
        if snapshot["projection_skipped"]:
            mutation_report["projection_skipped"] += 1

        if step % output_interval == 0:
            snapshots.append((step, snapshot))

    log_mutation_summary(mutation_report)
    return snapshots
