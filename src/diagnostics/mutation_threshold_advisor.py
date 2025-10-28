# src/diagnostics/mutation_threshold_advisor.py
# 📐 Mutation Threshold Advisor — computes adaptive pressure delta thresholds
# for mutation detection
# 📌 This module operates on per-cell simulation metadata.
# It does NOT interact with fluid_mask or geometry masking logic.
# It is NOT responsible for solver inclusion/exclusion decisions.

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def get_delta_threshold(cell, context):
    """
    Suggests an adaptive pressure delta threshold for mutation detection.

    Roadmap Alignment:
    Reflex Scoring:
    - Threshold adapts to resolution, divergence, timestep, and reflex score
    - Supports mutation gating and suppression fallback

    Parameters:
    - cell: Cell object containing position, velocity, and other properties
    - context: dict containing simulation metadata (e.g., resolution, divergence,
      time_step, reflex_score)

    Returns:
    - float: Recommended threshold for mutation detection
    """
    def scale_threshold(base, factor, condition):
        return base * factor if condition else base

    base_threshold = 1e-8

    resolution = context.get("resolution", "normal")
    base_threshold = scale_threshold(base_threshold, 0.1, resolution == "high")
    base_threshold = scale_threshold(base_threshold, 5, resolution == "low")

    local_divergence = context.get("divergence", 0.0)
    base_threshold = scale_threshold(base_threshold, 2, abs(local_divergence) < 0.01)
    base_threshold = scale_threshold(base_threshold, 0.5, abs(local_divergence) > 0.1)

    time_step = context.get("time_step", 0.05)
    base_threshold = scale_threshold(base_threshold, 2, time_step < 0.01)
    base_threshold = scale_threshold(base_threshold, 0.5, time_step > 0.2)

    reflex_score = context.get("reflex_score", 0.0)
    base_threshold = scale_threshold(base_threshold, 0.5, reflex_score < 0.2)
    base_threshold = scale_threshold(base_threshold, 1.5, reflex_score > 0.8)

    mutation_density = context.get("mutation_density", 0.0)
    base_threshold = scale_threshold(base_threshold, 0.75, mutation_density > 0.3)
    base_threshold = scale_threshold(base_threshold, 1.25, mutation_density < 0.05)

    final_threshold = max(base_threshold, 1e-15)

    if debug:
        print(
            f"[THRESHOLD] Computed delta threshold: {final_threshold:.2e} "
            f"(resolution={resolution}, divergence={local_divergence}, "
            f"dt={time_step}, score={reflex_score}, "
            f"density={mutation_density})"
        )

    return final_threshold
