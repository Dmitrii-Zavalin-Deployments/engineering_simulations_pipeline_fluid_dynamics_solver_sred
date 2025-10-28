# src/main_solver.py
# 🚀 Entry Point — Navier-Stokes Simulation Orchestrator
# 📌 This module coordinates input parsing, grid initialization, time integration,
# and reflex audit.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is
# geometry-mask-driven.

import os
import sys
import json
import yaml

from src.input_reader import load_simulation_input
from src.snapshot_manager import generate_snapshots
from src.compression.snapshot_compactor import compact_pressure_delta_map
from src.initialization.fluid_mask_initializer import build_simulation_grid
from src.config.config_validator import validate_config
from src.config.config_loader import load_simulation_config
from src.audit.run_reflex_audit import run_reflex_audit

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


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


def prepare_simulation_config(input_path: str) -> dict:
    input_data = load_simulation_input(input_path)
    ghost_rules_path = os.getenv("GHOST_RULES_PATH", "config/ghost_rules.json")
    if os.path.isfile(ghost_rules_path):
        with open(ghost_rules_path) as f:
            ghost_rules = json.load(f)
        input_data["ghost_rules"] = ghost_rules
        if debug:
            print(f"👻 Injected ghost_rules from: {ghost_rules_path}")
    return load_simulation_config(
        domain_path=input_path,
        ghost_path=ghost_rules_path,
        step_index=0
    )


def write_snapshot(output_folder: str, scenario: str, step: int, snapshot: dict):
    formatted_step = f"{step:04d}"
    filename = f"{scenario}_step_{formatted_step}.json"
    path = os.path.join(output_folder, filename)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    if debug:
        print(f"🔄 Step {formatted_step} written → {filename}")


def maybe_compact_pressure(step: int, score: float):
    if isinstance(score, (int, float)) and score >= 4:
        original = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
        compacted = f"data/snapshots/compacted/pressure_delta_compact_step_{step:04d}.json"
        compact_pressure_delta_map(original, compacted)
        if debug:
            print(f"📉 Compacted pressure delta map for step {step:04d}")


def run_navier_stokes_simulation(input_path: str, output_dir: str | None = None):
    scenario = os.path.splitext(os.path.basename(input_path))[0]
    config = prepare_simulation_config(input_path)
    output_folder = output_dir or os.path.join(
        "data", "testing-input-output", "navier_stokes_output"
    )
    os.makedirs(output_folder, exist_ok=True)

    reflex_config_path = os.getenv(
        "REFLEX_CONFIG", "config/reflex_debug_config.yaml"
    )
    reflex_config = load_reflex_config(reflex_config_path)

    if debug:
        ghost_cfg = config.get("ghost_rules", {})
        print(
            f"👻 Ghost Rules → Faces: {ghost_cfg.get('boundary_faces', [])}, "
            f"Default: {ghost_cfg.get('default_type')}"
        )
        print(f"   Face Types: {ghost_cfg.get('face_types', {})}")
        print(f"🧠 [main_solver] Starting Navier-Stokes simulation for: {scenario}")
        print(f"📄 Input path: {input_path}")
        print(f"📁 Output folder: {output_folder}")
        print(f"⚙️ Reflex config path: {reflex_config_path}")
        print("🛠️ Debug mode enabled.")
        print(f"📦 Input preview (truncated): {json.dumps(config, indent=2)[:1000]}")
        domain = config.get("domain_definition", {})
        print(
            f"📐 Grid resolution: {domain.get('nx')} × {domain.get('ny')} × "
            f"{domain.get('nz')}"
        )

    validate_config(config)
    build_simulation_grid(config)
    snapshots = generate_snapshots(config, scenario, config=reflex_config)

    for step, snapshot in snapshots:
        write_snapshot(output_folder, scenario, step, snapshot)
        maybe_compact_pressure(step, snapshot.get("reflex_score", 0.0))

    if debug:
        print(f"✅ Simulation complete. Total snapshots: {len(snapshots)}")

    run_reflex_audit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Navier-Stokes simulation and generate snapshots."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to simulation input file."
    )
    args = parser.parse_args()

    run_navier_stokes_simulation(args.input_file)
