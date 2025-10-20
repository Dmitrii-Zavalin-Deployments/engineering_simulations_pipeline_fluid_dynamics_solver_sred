# src/main_solver.py
# 🚀 Entry Point — Navier-Stokes Simulation Orchestrator

import os
import sys
import json
import yaml
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_reader import load_simulation_input
from src.snapshot_manager import generate_snapshots
from src.compression.snapshot_compactor import compact_pressure_delta_map
from src.metrics.reflex_score_evaluator import batch_evaluate_trace
from src.initialization.fluid_mask_initializer import build_simulation_grid  # ✅ Added

# ✅ Reflex config loader
def load_reflex_config(path="config/reflex_debug_config.yaml"):
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {
            "reflex_verbosity": "medium",
            "include_divergence_delta": False,
            "include_pressure_mutation_map": False,
            "log_projection_trace": False,
            "ghost_adjacency_depth": 1
        }

# ✅ Simulation runner — orchestrates full Navier-Stokes solve
def run_navier_stokes_simulation(input_path: str, output_dir: str | None = None, debug: bool = False):
    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    input_data = load_simulation_input(input_path)

    output_folder = output_dir or os.path.join("data", "testing-input-output", "navier_stokes_output")
    os.makedirs(output_folder, exist_ok=True)

    reflex_config_path = os.getenv("REFLEX_CONFIG", "config/reflex_debug_config.yaml")
    reflex_config = load_reflex_config(reflex_config_path)

    # ✅ Build reflex-tagged grid
    grid = build_simulation_grid(input_data)

    # 🧠 Simulation metadata
    print(f"🧠 [main_solver] Starting Navier-Stokes simulation for: {scenario_name}")
    print(f"📄 Input path: {input_path}")
    print(f"📁 Output folder: {output_folder}")
    print(f"⚙️  Reflex config path: {reflex_config_path}")

    if debug:
        print("🛠️ Debug mode enabled.")
        print(f"📦 Input preview (truncated): {json.dumps(input_data, indent=2)[:1000]}")
        domain = input_data.get("domain_definition", {})
        print(f"📐 Grid resolution: {domain.get('nx')} × {domain.get('ny')} × {domain.get('nz')}")

    # 🔁 Time Integration Loop — solves Navier-Stokes equations per step
    # 🧠 Roadmap Alignment:
    # - Structured input parsing → input_reader.py
    # - Momentum update: ρ(∂u/∂t + u · ∇u) = μ∇²u → momentum_solver.py
    #     - Advection: u · ∇u → advection.py
    #     - Viscosity: μ∇²u → viscosity.py
    # - Pressure solve: ∇²P = ∇ · u → pressure_solver.py
    # - Continuity enforcement: ∇ · u = 0 → velocity_projection.py
    # - Boundary enforcement → boundary_condition_solver.py
    # - Ghost logic → ghost_cell_generator.py, ghost_influence_applier.py
    # - Time loop orchestration → step_controller.py
    # - Output and diagnostics → snapshot_manager.py, reflex_controller.py

    snapshots = generate_snapshots(input_data, scenario_name, config=reflex_config)

    for step, snapshot in snapshots:
        formatted_step = f"{step:04d}"
        filename = f"{scenario_name}_step_{formatted_step}.json"
        path = os.path.join(output_folder, filename)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"🔄 Step {formatted_step} written → {filename}")

        # 📉 Optional compaction of pressure delta map
        score = snapshot.get("reflex_score", 0.0)
        if isinstance(score, (int, float)) and score >= 4:
            original_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
            compacted_path = f"data/snapshots/compacted/pressure_delta_compact_step_{step:04d}.json"
            compact_pressure_delta_map(original_path, compacted_path)
            if debug:
                print(f"📉 Compacted pressure delta map for step {formatted_step}")

    print(f"✅ Simulation complete. Total snapshots: {len(snapshots)}")

    # 📋 Reflex Audit — post-process integrity checks
    trace_dir = "data/snapshots"
    pathway_log = os.path.join(output_folder, "mutation_pathways_log.json")
    reflex_snapshots = [snap for (_, snap) in snapshots]

    audit_report = batch_evaluate_trace(trace_dir, pathway_log, reflex_snapshots)
    print(f"\n📋 Reflex Snapshot Audit:")
    for entry in audit_report:
        print(f"[AUDIT] Step {entry['step_index']:04d} → "
              f"Mutations={entry['mutated_cells']}, "
              f"Pathway={'✓' if entry['pathway_recorded'] else '✗'}, "
              f"Projection={'✓' if entry['has_projection'] else '✗'}, "
              f"Score={entry['reflex_score']}")

# ✅ CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Navier-Stokes simulation and generate snapshots.")
    parser.add_argument("input_file", type=str, help="Path to simulation input file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")
    args = parser.parse_args()

    run_navier_stokes_simulation(args.input_file, debug=args.debug)



