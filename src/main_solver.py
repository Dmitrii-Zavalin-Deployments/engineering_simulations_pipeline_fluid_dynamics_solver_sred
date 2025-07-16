# src/main_solver.py
# 🧠 Top-level driver for fluid simulation — orchestrates snapshots using step evolution

import os
import sys
import json
import logging
import yaml
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_reader import load_simulation_input
from src.snapshot_manager import generate_snapshots
from src.compression.snapshot_compactor import compact_pressure_delta_map  # ✅ Patch: import compactor

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

def run_solver(input_path: str, reflex_score_min: int = 0):
    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    input_data = load_simulation_input(input_path)
    output_folder = os.path.join("data", "testing-input-output", "navier_stokes_output")
    os.makedirs(output_folder, exist_ok=True)

    reflex_config_path = os.getenv("REFLEX_CONFIG", "config/reflex_debug_config.yaml")
    reflex_config = load_reflex_config(reflex_config_path)

    print(f"🧠 [main_solver] Starting simulation for: {scenario_name}")
    print(f"📄 Input path: {input_path}")
    print(f"📁 Output folder: {output_folder}")
    print(f"⚙️  Reflex config path: {reflex_config_path}")
    print(f"📊 Reflex compaction threshold: {reflex_score_min}")

    snapshots = generate_snapshots(input_data, scenario_name, config=reflex_config)

    for step, snapshot in snapshots:
        formatted_step = f"{step:04d}"
        filename = f"{scenario_name}_step_{formatted_step}.json"
        path = os.path.join(output_folder, filename)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"🔄 Step {formatted_step} written → {filename}")

        # ✅ Patch: trigger compaction for reflex-complete steps
        reflex_score = snapshot.get("reflex_score", 0)
        if reflex_score >= reflex_score_min:
            original = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
            compacted = f"data/snapshots/compacted/pressure_delta_compact_step_{step:04d}.json"
            compact_pressure_delta_map(original, compacted)

    print(f"✅ Simulation complete. Total snapshots: {len(snapshots)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fluid simulation and generate snapshots.")
    parser.add_argument("input_file", type=str, help="Path to simulation input file.")
    parser.add_argument("--reflex_score_min", type=int, default=0,
                        help="Minimum reflex score required to trigger compaction.")
    args = parser.parse_args()

    run_solver(args.input_file, reflex_score_min=args.reflex_score_min)



