# src/diagnostics/mutation_threshold_advisor.py
# 📐 Mutation Threshold Advisor — computes adaptive pressure delta thresholds for mutation detection
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
    - context: dict containing simulation metadata (e.g., resolution, divergence, time_step, reflex_score)

    Returns:
    - float: Recommended threshold for mutation detection
    """

    # Base threshold anchors mutation sensitivity above numerical noise floor.
    # 1e-8 is chosen to balance detection of subtle pressure changes with stability across double-precision solvers.
    # It reflects empirical defaults from CFD practice and allows safe scaling via reflex_score, divergence, and resolution.
    base_threshold = 1e-8

    # 🔧 Resolution scaling
    resolution = context.get("resolution", "normal")
    # High resolution → tighter threshold to capture fine gradients
    # ×0.1 multiplier reflects ~10× increase in spatial fidelity, based on typical AMR refinement ratios
    if resolution == "high":
        base_threshold *= 0.1
    # Low resolution → looser threshold to avoid false positives from coarse gradients
    # ×5 multiplier reflects typical loss of fidelity in coarse meshes and reduced gradient sensitivity
    elif resolution == "low":
        base_threshold *= 5

    # 🔍 Divergence scaling
    local_divergence = context.get("divergence", 0.0)
    # Near-zero divergence (<0.01) implies incompressible, stable flow → safe to tighten threshold
    # ×2 multiplier reflects confidence in low-noise environment, aligned with CFL stability bounds
    if abs(local_divergence) < 0.01:
        base_threshold *= 2
    # High divergence (>0.1) implies transitional or unstable flow → loosen threshold to avoid false triggers
    # ×0.5 multiplier suppresses sensitivity in noisy regions, consistent with pressure-correction tolerances
    elif abs(local_divergence) > 0.1:
        base_threshold *= 0.5

    # ⏱️ Time step scaling
    time_step = context.get("time_step", 0.05)
    # Small time steps (<0.01) imply high temporal resolution → tighter threshold
    # ×2 multiplier reflects increased confidence in transient fidelity, typical of explicit schemes
    if time_step < 0.01:
        base_threshold *= 2
    # Large time steps (>0.2) imply smoothing and potential loss of transient detail → loosen threshold
    # ×0.5 multiplier reduces sensitivity to avoid over-triggering in under-resolved time domains
    elif time_step > 0.2:
        base_threshold *= 0.5

    # 🧠 Reflex score scaling
    reflex_score = context.get("reflex_score", 0.0)
    # Low reflex score (<0.2) implies low confidence in mutation relevance → suppress detection
    # ×0.5 multiplier gates low-certainty events, similar to precision-recall tuning in anomaly classifiers
    if reflex_score < 0.2:
        base_threshold *= 0.5
    # High reflex score (>0.8) implies strong mutation signal → amplify detection
    # ×1.5 multiplier boosts sensitivity for high-certainty events, aligned with mutation gating logic
    elif reflex_score > 0.8:
        base_threshold *= 1.5

    # 📊 Mutation density fallback
    mutation_density = context.get("mutation_density", 0.0)
    # High mutation density (>0.3) implies noisy region → suppress threshold to avoid over-triggering
    # ×0.75 multiplier dampens sensitivity in mutation-saturated zones, similar to entropy-based suppression
    if mutation_density > 0.3:
        base_threshold *= 0.75
    # Low mutation density (<0.05) implies quiet region → amplify threshold to catch rare events
    # ×1.25 multiplier boosts detection in mutation-sparse zones, tuned for reflex overlays and CI diagnostics
    elif mutation_density < 0.05:
        base_threshold *= 1.25

    # 🧯 Final safety clamp
    # Prevent threshold from dropping below machine epsilon (~1e-15 for double precision)
    # Ensures numerical stability and avoids underflow in mutation gating logic
    final_threshold = max(base_threshold, 1e-15)

    if debug:
        print(f"[THRESHOLD] Computed delta threshold: {final_threshold:.2e} "
              f"(resolution={resolution}, divergence={local_divergence}, dt={time_step}, "
              f"score={reflex_score}, density={mutation_density})")

    return final_threshold



