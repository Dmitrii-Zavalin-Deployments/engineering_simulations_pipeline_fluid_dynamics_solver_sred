# src/main_solver.py

import os
import json
from datetime import datetime

def run_solver(input_path: str = None, output_base: str = None):
    """
    Executes simulation pipeline. If no input is given, triggers full upload preparation.
    Otherwise, writes a stub snapshot for the specified scenario input.
    """
    if input_path is None or output_base is None:
        print("🚀 [main_solver] Triggering full simulation pipeline")
        from prepare_simulation_upload import prepare_simulation_upload_files  # ⬅️ Local import to avoid circular reference
        prepare_simulation_upload_files()
        return

    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    scenario_output_dir = os.path.join(output_base, scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    print(f"🧠 [main_solver] Starting simulation for: {scenario_name}")
    print(f"📄 [main_solver] Input path: {input_path}")
    print(f"📁 [main_solver] Output folder: {scenario_output_dir}")

    # Stub snapshot content — to be replaced later with real simulation output
    snapshot = {
        "divergence_max": 0.0,
        "velocity_max": 0.0,
        "projection_passes": 1,
        "reflex_triggered": False,
        "overflow_flag": False,
        "volatility_slope": "flat",
        "volatility_delta": 0.0,
        "damping_applied": False,
        "step_index": 0,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    snapshot_path = os.path.join(scenario_output_dir, "step_0000.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"📦 [main_solver] Snapshot written: {snapshot_path}")


# 🚀 Entry point — triggers full pipeline when run as script
if __name__ == "__main__":
    run_solver()



