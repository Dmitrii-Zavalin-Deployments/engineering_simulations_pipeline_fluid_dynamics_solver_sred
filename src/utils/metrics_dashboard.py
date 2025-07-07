# metrics_dashboard.py

"""
📊 Metrics Dashboard for Runtime Visualization
Visualizes key fluid solver diagnostics:
  • Divergence (∇·u) histogram
  • Kinetic energy trajectory
  • Pressure range over time
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_divergence_logs(log_dir):
    """
    Loads all divergence log files into sorted list of metrics.

    Args:
        log_dir (str): Path to 'logs/' folder.

    Returns:
        List of dicts sorted by step number.
    """
    entries = []
    for fname in os.listdir(log_dir):
        if fname.startswith("divergence_step_") and fname.endswith(".json"):
            path = os.path.join(log_dir, fname)
            with open(path) as f:
                data = json.load(f)
                entries.append(data)
    return sorted(entries, key=lambda d: d["step"])


def plot_divergence_histogram(divergence_field, step):
    """
    Plots histogram of ∇·u at given step.

    Args:
        divergence_field (np.ndarray): Full divergence field [nx+2, ny+2, nz+2]
        step (int): Simulation step
    """
    interior = divergence_field[1:-1, 1:-1, 1:-1].flatten()
    interior = np.nan_to_num(interior, nan=0.0, posinf=0.0, neginf=0.0)

    plt.figure(figsize=(8, 4))
    plt.hist(interior, bins=100, color='steelblue', alpha=0.8)
    plt.title(f"∇·u Histogram @ Step {step}")
    plt.xlabel("Divergence Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ke_and_pressure_metrics(log_data):
    """
    Plots kinetic energy, pressure range, and mean pressure vs time.

    Args:
        log_data (List[dict]): Sequence of log entries.
    """
    steps = [entry["step"] for entry in log_data]
    ke = [entry.get("kinetic_energy", np.nan) for entry in log_data]
    mean_pressure = [entry.get("mean_pressure", np.nan) for entry in log_data]
    min_pressure = [entry.get("min_pressure", np.nan) for entry in log_data]
    max_pressure = [entry.get("max_pressure", np.nan) for entry in log_data]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, ke, label='Kinetic Energy', color='darkgreen')
    axes[0].set_ylabel("KE")
    axes[0].set_title("Kinetic Energy Trajectory")
    axes[0].grid(True)

    axes[1].plot(steps, mean_pressure, label='Mean Pressure', color='navy')
    axes[1].set_ylabel("Pressure")
    axes[1].set_title("Mean Pressure Over Time")
    axes[1].grid(True)

    axes[2].fill_between(steps, min_pressure, max_pressure, color='orange', alpha=0.4)
    axes[2].set_ylabel("Pressure Range")
    axes[2].set_title("Pressure Min–Max Envelope")
    axes[2].grid(True)
    axes[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def visualize_metrics(output_dir):
    """
    Entry point to load logs and generate dashboard plots.

    Args:
        output_dir (str): Root simulation output directory
    """
    log_dir = os.path.join(output_dir, "logs")

    if not os.path.isdir(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        return

    log_entries = load_divergence_logs(log_dir)

    if not log_entries:
        print("⚠️ No divergence log entries found.")
        return

    print(f"📊 Loaded {len(log_entries)} log entries from: {log_dir}")

    plot_ke_and_pressure_metrics(log_entries)

    # Optional: plot histogram from last step
    final_step = log_entries[-1]["step"]
    field_path = os.path.join(output_dir, "fields", f"step_{final_step:04d}.json")
    if os.path.exists(field_path):
        with open(field_path) as f:
            snapshot = json.load(f)
        div = np.array(snapshot.get("divergence", []))
        if div.ndim == 3:
            plot_divergence_histogram(div, final_step)
        else:
            print("⚠️ No valid divergence field in snapshot.")
    else:
        print("ℹ️ No snapshot found for final step — skipping histogram.")



