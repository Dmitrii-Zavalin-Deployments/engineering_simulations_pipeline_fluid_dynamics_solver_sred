# tests/utils/validators/trace_sanity_checker.py
# 🧪 Trace Snapshot Auditor — reflex alignment & consistency checks

def validate_trace(snapshot: dict):
    """
    Asserts core consistency between ghost influence and pressure mutation logic.
    Args:
        snapshot (dict): Dictionary containing reflex metrics from snapshot
    Raises:
        AssertionError if any check fails
    """
    step = snapshot.get("step_index", "unknown")

    pressure_mutation_count = snapshot.get("pressure_mutation_count")
    influenced_by_ghost_count = snapshot.get("influenced_by_ghost_count")
    pre_divergence = snapshot.get("pre_divergence")
    post_divergence = snapshot.get("post_divergence")

    assert isinstance(pressure_mutation_count, int), f"[step {step}] Invalid pressure_mutation_count"
    assert isinstance(influenced_by_ghost_count, int), f"[step {step}] Invalid influenced_by_ghost_count"
    assert isinstance(pre_divergence, float), f"[step {step}] Invalid pre_divergence"
    assert isinstance(post_divergence, float), f"[step {step}] Invalid post_divergence"

    assert pressure_mutation_count >= influenced_by_ghost_count, (
        f"[step {step}] Ghost-influenced cells exceed pressure mutations — trace mismatch"
    )
    assert post_divergence < pre_divergence + 1e-6, (
        f"[step {step}] Divergence did not decrease after projection: pre={pre_divergence}, post={post_divergence}"
    )

    print(f"[TRACE CHECK] Step {step} → ✅ Passed: divergence collapsed, mutation trace consistent.")


def validate_trace_file(path: str):
    """
    Loads snapshot JSON trace and applies validation.
    Args:
        path (str): Path to snapshot file
    """
    import json
    with open(path, 'r') as file:
        snapshot = json.load(file)
    validate_trace(snapshot)



