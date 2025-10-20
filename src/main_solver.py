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
from src.initialization.fluid_mask_initializer import build_simulation_grid
from src.config.config_validator import validate_config
from src.audit.run_reflex_audit import run_reflex_audit  # ✅ Added

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

    # ✅ Ghost rule pre-check and logging
    ghost_cfg = input_data.get("ghost_rules", {})
    if debug:
        print(f"👻 Ghost Rules → Faces: {ghost_cfg.get('boundary_faces', [])}, Default: {ghost_cfg.get('default_type')}")
        print(f"   Face Types: {ghost_cfg.get('face_types', {})}")

    # ✅ Validate config before grid setup
    validate_config(input_data)

    # ✅ Build reflex-tagged grid
    build_simulation_grid(input_data)

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

    # 📋 Reflex Audit — bundled scoring, overlays, and integrity panel
    run_reflex_audit()

# ✅ CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Navier-Stokes simulation and generate snapshots.")
    parser.add_argument("input_file", type=str, help="Path to simulation input file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")
    args = parser.parse_args()

    run_navier_stokes_simulation(args.input_file, debug=args.debug)



