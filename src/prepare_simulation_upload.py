# src/prepare_simulation_upload.py

import os
import glob
from main_solver import run_solver

BASE_DIR = "data/testing-input-output"
OUTPUT_SUBDIR = "navier_stokes_output"
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_SUBDIR)

def prepare_simulation_upload_files():
    print(f"📦 Preparing simulation upload folder: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = glob.glob(os.path.join(BASE_DIR, "*.json"))
    if not input_files:
        print("⚠️ No input files found in testing-input-output/. Aborting.")
        return

    for input_path in input_files:
        scenario = os.path.basename(input_path)
        print(f"🚀 Processing scenario: {scenario}")
        try:
            run_solver(input_path, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Error during simulation for {scenario}: {e}")
        else:
            print(f"✅ Scenario snapshot saved: {os.path.join(OUTPUT_DIR, scenario.replace('.json', ''))}")

    print("🎯 All simulation upload files are ready.")



