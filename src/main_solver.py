# src/main_solver.py

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

def run_solver(input_path: str, output_dir: str | None = None, debug: bool = False):  # ✅ Added debug flag
    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    input_data = load_simulation_input(input_path)

    output_folder = output_dir or os.path.join("data", "testing-input-output", "navier_stokes_output")
    os.makedirs(output_folder, exist_ok=True)

    reflex_config_path = os.getenv("REFLEX_CONFIG", "config/reflex_debug_config.yaml")
    reflex_config = load_reflex_config(reflex_config_path)

    print(f"🧠 [main_solver] Starting simulation for: {scenario_name}")  # Debug: scenario name
    print(f"📄 Input path: {input_path}")  # Debug: input file path
    print(f"📁 Output folder: {output_folder}")  # Debug: output directory
    print(f"⚙️  Reflex config path: {reflex_config_path}")  # Debug: config file path

    if debug:
        print("🛠️ Debug mode enabled.")  # Debug: flag confirmation
        print(f"📦 Input preview (truncated): {json.dumps(input_data, indent=2)[:1000]}")  # Debug: input snapshot
        domain = input_data.get("domain_definition", {})
        print(f"📐 Grid resolution: {domain.get('nx')} × {domain.get('ny')} × {domain.get('nz')}")  # Debug: grid size

    snapshots = generate_snapshots(input_data, scenario_name, config=reflex_config)

    for step, snapshot in snapshots:
        formatted_step = f"{step:04d}"
        filename = f"{scenario_name}_step_{formatted_step}.json"
        path = os.path.join(output_folder, filename)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"🔄 Step {formatted_step} written → {filename}")  # Debug: snapshot write confirmation

        score = snapshot.get("reflex_score")
        if not isinstance(score, (int, float)):
            score = 0.0
        if score >= 4:
            original_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
            compacted_path = f"data/snapshots/compacted/pressure_delta_compact_step_{step:04d}.json"
            compact_pressure_delta_map(original_path, compacted_path)
            if debug:
                print(f"📉 Compacted pressure delta map for step {formatted_step}")  # Debug: compaction trigger

    print(f"✅ Simulation complete. Total snapshots: {len(snapshots)}")  # Debug: completion summary

    trace_dir = "data/snapshots"
    pathway_log = os.path.join(output_folder, "mutation_pathways_log.json")
    reflex_snapshots = [snap for (_, snap) in snapshots]

    audit_report = batch_evaluate_trace(trace_dir, pathway_log, reflex_snapshots)
    print(f"\n📋 Reflex Snapshot Audit:")  # Debug: audit header
    for entry in audit_report:
        print(f"[AUDIT] Step {entry['step_index']:04d} → "
              f"Mutations={entry['mutated_cells']}, "
              f"Pathway={'✓' if entry['pathway_recorded'] else '✗'}, "
              f"Projection={'✓' if entry['has_projection'] else '✗'}, "
              f"Score={entry['reflex_score']}")  # Debug: audit entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fluid simulation and generate snapshots.")
    parser.add_argument("input_file", type=str, help="Path to simulation input file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")  # ✅ Added CLI flag
    args = parser.parse_args()

    run_solver(args.input_file, debug=args.debug)



