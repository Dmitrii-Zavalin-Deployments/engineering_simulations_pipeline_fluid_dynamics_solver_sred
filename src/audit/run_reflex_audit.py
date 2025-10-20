# src/audit/run_reflex_audit.py
# 📋 Reflex Audit Runner — bundles scoring, overlays, and dashboard export for simulation integrity review

import os
import json
from src.metrics.reflex_score_evaluator import batch_evaluate_trace
from src.visualization.overlay_integrity_panel import render_integrity_panel
from src.reflex.reflex_pathway_logger import log_reflex_pathway

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

    print(f"📋 Starting Reflex Audit...")
    snapshots = load_snapshots(snapshot_dir)
    if not snapshots:
        print("[AUDIT] No snapshots found.")
        return

    trace_report = batch_evaluate_trace(snapshot_dir, pathway_log, snapshots)

    print(f"\n📊 Reflex Snapshot Summary:")
    for entry in trace_report:
        print(f"[AUDIT] Step {entry['step_index']:04d} → "
              f"Mutations={entry['mutated_cells']}, "
              f"Pathway={'✓' if entry['pathway_recorded'] else '✗'}, "
              f"Projection={'✓' if entry['has_projection'] else '✗'}, "
              f"Score={entry['reflex_score']}")

    print(f"\n🖼️ Generating Overlay Integrity Panel...")
    render_integrity_panel(snapshot_dir=snapshot_dir, output_path=os.path.join(output_folder, "integrity_panel.png"))

    print(f"\n✅ Reflex Audit Complete. Outputs saved to → {output_folder}")

# ✅ CLI entrypoint
if __name__ == "__main__":
    run_reflex_audit()



