# src/main_solver.py
# 🚀 Entry Point — Navier-Stokes Simulation Orchestrator
# 📌 This module coordinates input parsing, grid initialization, time integration, and reflex audit.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import os
import sys
import json
import yaml

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_reader import load_simulation_input
from src.snapshot_manager import generate_snapshots
from src.compression.snapshot_compactor import compact_pressure_delta_map
from src.initialization.fluid_mask_initializer import build_simulation_grid
from src.config.config_validator import validate_config
from src.audit.run_reflex_audit import run_reflex_audit

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

def run_navier_stokes_simulation(input_path: str, output_dir: str | None = None):
    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    input_data = load_simulation_input(input_path)

    ghost_rules_path = os.getenv("GHOST_RULES_PATH")
    if ghost_rules_path and os.path.isfile(ghost_rules_path):
        with open(ghost_rules_path) as f:
            ghost_rules = json.load(f)
        input_data["ghost_rules"] = ghost_rules
        if debug:
            print(f"👻 Injected ghost_rules from: {ghost_rules_path}")

    output_folder = output_dir or os.path.join("data", "testing-input-output", "navier_stokes_output")
    os.makedirs(output_folder, exist_ok=True)

    reflex_config_path = os.getenv("REFLEX_CONFIG", "config/reflex_debug_config.yaml")
    reflex_config = load_reflex_config(reflex_config_path)

    ghost_cfg = input_data.get("ghost_rules", {})
    if debug:
        print(f"👻 Ghost Rules → Faces: {ghost_cfg.get('boundary_faces', [])}, Default: {ghost_cfg.get('default_type')}")
        print(f"   Face Types: {ghost_cfg.get('face_types', {})}")

    validate_config(input_data)
    build_simulation_grid(input_data)

    if debug:
        print(f"🧠 [main_solver] Starting Navier-Stokes simulation for: {scenario_name}")
        print(f"📄 Input path: {input_path}")
        print(f"📁 Output folder: {output_folder}")
        print(f"⚙️ Reflex config path: {reflex_config_path}")
        print("🛠️ Debug mode enabled.")
        print(f"📦 Input preview (truncated): {json.dumps(input_data, indent=2)[:1000]}")
        domain = input_data.get("domain_definition", {})
        print(f"📐 Grid resolution: {domain.get('nx')} × {domain.get('ny')} × {domain.get('nz')}")

    snapshots = generate_snapshots(input_data, scenario_name, config=reflex_config)

    for step, snapshot in snapshots:
        formatted_step = f"{step:04d}"
        filename = f"{scenario_name}_step_{formatted_step}.json"
        path = os.path.join(output_folder, filename)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        if debug:
            print(f"🔄 Step {formatted_step} written → {filename}")

        score = snapshot.get("reflex_score", 0.0)
        if isinstance(score, (int, float)) and score >= 4:
            original_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
            compacted_path = f"data/snapshots/compacted/pressure_delta_compact_step_{step:04d}.json"
            compact_pressure_delta_map(original_path, compacted_path)
            if debug:
                print(f"📉 Compacted pressure delta map for step {formatted_step}")

    if debug:
        print(f"✅ Simulation complete. Total snapshots: {len(snapshots)}")

    run_reflex_audit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Navier-Stokes simulation and generate snapshots.")
    parser.add_argument("input_file", type=str, help="Path to simulation input file.")
    args = parser.parse_args()

    run_navier_stokes_simulation(args.input_file)



