# src/audit/run_reflex_audit.py
# 📋 Reflex Audit Runner — bundles scoring, overlays, and dashboard export for simulation integrity review
# 📌 This module operates entirely on exported snapshot data.
# It does NOT interact with fluid_mask or geometry masking logic.
# It is NOT responsible for solver inclusion/exclusion decisions.

import os
import json
from src.metrics.reflex_score_evaluator import batch_evaluate_trace
from src.visualization.overlay_integrity_panel import render_integrity_panel

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def load_snapshots(snapshot_dir: str) -> list:
    """
    Loads snapshot JSON files from a directory.

    Args:
        snapshot_dir (str): Path to snapshot directory

    Returns:
        List[Dict]: List of snapshot dictionaries
    """
    snapshots = []
    for fname in sorted(os.listdir(snapshot_dir)):
        if fname.endswith(".json") and "step_" in fname:
            path = os.path.join(snapshot_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                    snapshots.append(data)
            except Exception as e:
                if debug:
                    print(f"[ERROR] Failed to load {fname}: {e}")
    return snapshots


def run_reflex_audit(
    snapshot_dir: str = "data/snapshots",
    output_folder: str = "data/diagnostics",
    pathway_log: str = "data/diagnostics/mutation_pathways_log.json"
):
    """
    Runs reflex audit across snapshots and generates scoring report and overlay panel.

    Args:
        snapshot_dir (str): Directory containing snapshot JSON files
        output_folder (str): Directory to store audit outputs
        pathway_log (str): Path to mutation pathway log file
    """
    os.makedirs(output_folder, exist_ok=True)

    if debug:
        print("📋 Starting Reflex Audit...")

    snapshots = load_snapshots(snapshot_dir)
    if not snapshots:
        if debug:
            print("[AUDIT] No snapshots found.")
        return

    trace_report = batch_evaluate_trace(snapshot_dir, pathway_log, snapshots)

    if debug:
        print("\n📊 Reflex Snapshot Summary:")
        for entry in trace_report:
            print(
                f"[AUDIT] Step {entry['step_index']:04d} → "
                f"Mutations={entry['mutated_cells']}, "
                f"Pathway={'✓' if entry['pathway_recorded'] else '✗'}, "
                f"Projection={'✓' if entry['has_projection'] else '✗'}, "
                f"Score={entry['reflex_score']}"
            )

        print("\n🖼️ Generating Overlay Integrity Panel...")

    render_integrity_panel(
        snapshot_dir=snapshot_dir,
        output_path=os.path.join(output_folder, "integrity_panel.png")
    )

    if debug:
        print(f"\n✅ Reflex Audit Complete. Outputs saved to → {output_folder}")


# ✅ CLI entrypoint
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Reflex Audit on simulation snapshots."
    )
    parser.add_argument(
        "--snapshot_dir",
        default="data/snapshots",
        help="Directory containing snapshot JSON files"
    )
    parser.add_argument(
        "--output_folder",
        default="data/diagnostics",
        help="Directory to store audit outputs"
    )
    parser.add_argument(
        "--pathway_log",
        default="data/diagnostics/mutation_pathways_log.json",
        help="Path to mutation pathway log file"
    )
    args = parser.parse_args()

    run_reflex_audit(
        snapshot_dir=args.snapshot_dir,
        output_folder=args.output_folder,
        pathway_log=args.pathway_log
    )



