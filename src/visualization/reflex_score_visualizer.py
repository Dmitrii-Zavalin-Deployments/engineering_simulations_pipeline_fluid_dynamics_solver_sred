# src/visualization/reflex_score_visualizer.py
# 📈 Reflex Score Visualizer — charts score evolution and mutation triggers across simulation steps

import matplotlib.pyplot as plt
import os
from typing import List, Dict

def plot_reflex_score_evolution(
    evaluations: List[Dict],
    output_path: str = "data/plots/reflex_score_evolution.png",
    title: str = "Reflex Score Evolution"
):
    """
    Plots reflex score evolution across steps with mutation triggers and suppression context.

    Args:
        evaluations (List[Dict]): List of per-step evaluation dicts from batch_evaluate_trace()
        output_path (str): Path to save the plot
        title (str): Plot title
    """
    if not evaluations:
        print("⚠️ No evaluation data provided.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    steps = [e["step_index"] for e in evaluations]
    scores = [e["reflex_score"] for e in evaluations]
    suppressions = [e["suppression_zone_count"] for e in evaluations]
    adjacents = [e["adjacency_count"] for e in evaluations]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(steps, scores, label="Reflex Score", color="blue", linewidth=2, marker="o")
    ax1.set_xlabel("Step Index")
    ax1.set_ylabel("Reflex Score", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title(title)

    # Overlay suppression zones
    ax2 = ax1.twinx()
    ax2.plot(steps, suppressions, label="Suppression Zones", color="gray", linestyle="--", marker="x")
    ax2.set_ylabel("Suppression Zone Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Optional: annotate adjacency counts
    for i, step in enumerate(steps):
        ax1.annotate(f"A:{adjacents[i]}", (step, scores[i]), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)

    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"📈 Reflex score plot saved → {output_path}")



