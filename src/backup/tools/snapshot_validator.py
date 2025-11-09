# src/tools/snapshot_validator.py
# 🧪 Snapshot Validator — cross-checks reflex flags, influence logs, and divergence collapse per step
# 📌 This module validates reflex causality, ghost influence tagging, and divergence collapse integrity.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import os
import json

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def load_snapshots(folder_path):
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".json") and "_step_" in f
    ])

def load_influence_log(folder_path):
    path = os.path.join(folder_path, "influence_flags_log.json")
    return json.load(open(path)) if os.path.exists(path) else []

def validate_pressure_mutation(reflex, step_index):
    causes = reflex.get("triggered_by", [])
    mutated = reflex.get("pressure_mutated", False)
    projected = reflex.get("velocity_projected", False)

    if mutated and "ghost_influence" not in causes and "boundary_override" not in causes:
        if debug:
            print(f"⚠️ [Step {step_index}] Pressure mutated without ghost or boundary trigger.")
    if not mutated and projected:
        if debug:
            print(f"⚠️ [Step {step_index}] Velocity projected but no pressure mutation flagged.")

def validate_influence_consistency(snapshot, influence_log, step_index):
    tag_count = sum(
        1 for c in snapshot.get("grid", [])
        if c.get("fluid_mask", False) and c.get("influenced_by_ghost", False)
    )
    entry = next((e for e in influence_log if e["step_index"] == step_index), {})
    exported_count = entry.get("influenced_cell_count", 0)

    if tag_count != exported_count:
        if debug:
            print(f"⚠️ [Step {step_index}] Influence flag mismatch → tagged:{tag_count}, exported:{exported_count}")

def validate_divergence_collapse(folder_path, step_index):
    log_path = os.path.join(folder_path, "divergence_log.txt")
    if not os.path.exists(log_path):
        if debug:
            print("⚠️ No divergence_log.txt found.")
        return

    with open(log_path) as f:
        lines = f.readlines()

    step_lines = [l for l in lines if f"Step {step_index:04d}" in l]
    if len(step_lines) < 2:
        if debug:
            print(f"⚠️ [Step {step_index}] Not enough divergence logs to verify collapse.")
        return

    pre_line = next((l for l in step_lines if "Stage: before projection" in l), "")
    post_line = next((l for l in step_lines if "Stage: after projection" in l), "")

    def extract_max(s):
        try:
            return float(s.split("Max:")[1].split("|")[0].strip())
        except:
            return None

    pre_div = extract_max(pre_line)
    post_div = extract_max(post_line)

    if pre_div is not None and post_div is not None:
        if post_div > pre_div:
            if debug:
                print(f"⚠️ [Step {step_index}] Divergence increased → pre:{pre_div:.2e}, post:{post_div:.2e}")
        elif post_div < pre_div:
            if debug:
                print(f"✅ [Step {step_index}] Divergence decreased → pre:{pre_div:.2e}, post:{post_div:.2e}")
        else:
            if debug:
                print(f"ℹ️ [Step {step_index}] Divergence unchanged → {post_div:.2e}")
    else:
        if debug:
            print(f"⚠️ [Step {step_index}] Failed to parse divergence values.")

def run_snapshot_validation(output_folder):
    if debug:
        print(f"🔍 Validating snapshots in: {output_folder}")
    snapshots = load_snapshots(output_folder)
    influence_log = load_influence_log(output_folder)

    for snap_path in snapshots:
        with open(snap_path) as f:
            snap = json.load(f)
        step_index = snap.get("step_index", -1)
        validate_pressure_mutation(snap, step_index)
        validate_influence_consistency(snap, influence_log, step_index)
        validate_divergence_collapse(output_folder, step_index)

    if debug:
        print("✅ Snapshot validation complete.")

if __name__ == "__main__":
    default = "data/testing-input-output/navier_stokes_output"
    run_snapshot_validation(default)



