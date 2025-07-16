# src/compression/snapshot_compactor.py
# 📦 Snapshot Compactor — removes unmutated cells from pressure delta map exports

import os
import json

def compact_pressure_delta_map(
    input_path: str,
    output_path: str,
    mutation_threshold: float = 1e-8
) -> int:
    """
    Compacts a pressure delta map by removing cells with negligible delta.

    Args:
        input_path (str): Path to original pressure delta map JSON
        output_path (str): Path to write compacted snapshot
        mutation_threshold (float): Minimum delta to retain a cell

    Returns:
        int: Number of cells retained after compaction
    """
    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[COMPACTOR] ❌ Failed to load {input_path}: {e}")
        return 0

    retained = {
        coord: info
        for coord, info in data.items()
        if abs(info.get("delta", 0.0)) > mutation_threshold
    }

    if not retained:
        print(f"[COMPACTOR] ⚠️ No cells retained after compaction — skipping write.")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(retained, f, indent=2)

    print(f"[COMPACTOR] ✅ Compacted snapshot saved → {output_path}")
    print(f"[COMPACTOR] 🧮 Cells retained: {len(retained)} of {len(data)}")
    return len(retained)



