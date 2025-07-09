# src/main_solver.py

import json
import os
from simulation.adaptive_scheduler import AdaptiveScheduler
from stability_utils import get_threshold

THRESHOLD_PATH = os.path.join("tests", "test_thresholds.json")

def load_env_variable(key, fallback=None):
    """Safely load .env variables with fallback"""
    return os.getenv(key, fallback)

def get_flat_scheduler_config(threshold_path):
    """Load damping_tests section and flatten it"""
    with open(threshold_path) as f:
        raw = json.load(f)
        return raw.get("damping_tests", {})

def main():
    # 🧩 Load environment mode if applicable
    simulation_mode = load_env_variable("SIMULATION_MODE", "default")
    print(f"[INFO] Simulation mode: {simulation_mode}")

    # 📦 Load reflex thresholds
    scheduler_config = get_flat_scheduler_config(THRESHOLD_PATH)

    # 🧠 Initialize Reflex-Aware Scheduler
    scheduler = AdaptiveScheduler(scheduler_config)

    # 🔧 Diagnostic output (optional)
    print(f"[INFO] Scheduler initialized with thresholds:")
    for k, v in scheduler_config.items():
        print(f"  {k}: {v}")

    # 🚧 Placeholder for main simulation loop or solver integration
    # e.g. run_simulation_step(scheduler), monitor volatility, etc.
    print("[STATUS] Simulation solver logic not yet implemented.")

if __name__ == "__main__":
    main()



