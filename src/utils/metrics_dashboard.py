# utils/metrics_dashboard.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_divergence_logs(log_dir):
    """
    Loads divergence JSON logs from specified directory.

    Returns:
        List of dictionaries sorted by step count.
    """
    entries = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.startswith("divergence_step_") and fname.endswith(".json"):
            path = os.path.join(log_dir, fname)
            with open(path) as f:
                data = json.load(f)
                entries.append(data)
    entries.sort(key=lambda x: x.get("step", 0))
    return entries

def plot_ke_and_pressure(log_dir, output_path=None):
    """
    Plots kinetic energy and pressure range trajectories.

    Requires corresponding log entries to contain KE, pressure_range, etc.
    """
    entries = load_divergence_logs(log_dir)
    steps = [e["step"] for e in entries]
    ke = [e.get("kinetic_energy", None) for e in entries]
    pmin = [e.get("pressure_min", None) or e.get("pressure_range", [None])[0] for e in entries]
    pmax = [e.get("pressure_max", None) or e.get("pressure_range", [None])[1] for e in entries]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(steps, ke, label="Kinetic Energy", color="darkblue")
    ax[0].set_ylabel("KE")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(steps, pmin, label="Pressure Min", color="darkred")
    ax[1].plot(steps, pmax, label="Pressure Max", color="darkgreen")
    ax[1].set_ylabel("Pressure")
    ax[1].set_xlabel("Step")
    ax[1].grid()
    ax[1].legend()

    fig.suptitle("Kinetic Energy & Pressure Range")
    if output_path:
        plt.savefig(output_path, dpi=120)
        print(f"📊 Saved KE/Pressure plot to: {output_path}")
    else:
        plt.show()

def plot_divergence_histogram(divergence_field, step=None, output_path=None):
    """
    Plots histogram of ∇·u at given step.

    Args:
        divergence_field (np.ndarray): Full divergence field [nx+2, ny+2, nz+2]
        step (int): Optional label
        output_path (str): Save path if specified
    """
    interior = divergence_field[1:-1, 1:-1, 1:-1]
    interior = np.nan_to_num(interior, nan=0.0, posinf=0.0, neginf=0.0)
    flattened = interior.flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(flattened, bins=100, color="steelblue", alpha=0.85)
    plt.xlabel("∇·u")
    plt.ylabel("Frequency")
    title = f"Divergence Histogram"
    if step is not None:
        title += f" @ Step {step}"
    plt.title(title)
    plt.grid()

    if output_path:
        plt.savefig(output_path, dpi=120)
        print(f"📉 Saved divergence histogram to: {output_path}")
    else:
        plt.show()



