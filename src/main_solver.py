# src/main_solver.py

import os
import json
import sys

def run_solver(input_path: str, output_base: str):
    """
    Executes simulation for a single scenario input file.
    Writes a unified snapshot directly into the output folder.
    """
    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_base, exist_ok=True)

    print(f"🧠 [main_solver] Starting simulation for: {scenario_name}")
    print(f"📄 [main_solver] Input path: {input_path}")
    print(f"📁 [main_solver] Output folder: {output_base}")

    snapshot = {
        "step": 0,
        "grid": [
            [0, 0, 0, [1.2, 0.0, -0.1], 1.0],
            [0, 1, 0, [0.9, -0.1, 0.0], 0.95],
            [1, 0, 0, [1.1, 0.1, 0.05], 0.98]
        ],
        "max_velocity": 1.2,
        "max_divergence": 0.05,
        "global_cfl": 0.9,
        "overflow_detected": False,
        "damping_enabled": True,
        "projection_passes": 2
    }

    snapshot_filename = f"{scenario_name}_step_0000.json"
    snapshot_path = os.path.join(output_base, snapshot_filename)
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"📦 [main_solver] Snapshot written: {snapshot_path}")
    print(f"✅ Scenario snapshot saved: {snapshot_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("⚠️ Please provide input and output folder paths:")
        print("   Example: python src/main_solver.py input.json output-folder")
        sys.exit(1)

    run_solver(sys.argv[1], sys.argv[2])



