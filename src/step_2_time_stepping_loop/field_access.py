# src/step_2_time_stepping_loop/field_access.py
# 🧱 Step 2: Field Access — Build Neighbor Maps

from typing import Dict, Any

debug = True  # toggle to True for verbose GitHub Action logs

ORDER_6 = ("xp", "xm", "yp", "ym", "zp", "zm")


def build_neighbor_map(cell_dict: Dict[str, Any], timestep: int) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Build neighbor velocity maps for all cells in the domain.

    Parameters
    ----------
    cell_dict : dict
        Full dictionary of cells, keyed by flat_index (as string).
        Each cell contains:
          - "grid_index": [i, j, k]
          - "flat_index_*": neighbor indices (or None at boundaries)
          - "time_history": {timestep: {"velocity": {"vx","vy","vz"}, "pressure": float}}
    timestep : int
        Current timestep index to fetch velocities from time_history.

    Returns
    -------
    Dict[int, Dict[str, Dict[str, float]]]
        For each cell (flat_index as int), returns a dict:
        {
          "vx_neighbors": {"xp": float, "xm": float, "yp": float, "ym": float, "zp": float, "zm": float},
          "vy_neighbors": {...},
          "vz_neighbors": {...}
        }

    Notes
    -----
    - If a neighbor index is None (boundary), the value is clamped to the current cell’s velocity.
    - This produces neighbor sets for all three velocity components, ready for diffusion/advection stencils.
    - Pressure neighbors can be added similarly if needed.
    """
    neighbor_map: Dict[int, Dict[str, Dict[str, float]]] = {}

    if debug:
        print(f"🔍 Building neighbor map for timestep {timestep}... total cells: {len(cell_dict)}")

    for flat_idx_str, cell in cell_dict.items():
        flat_idx = int(flat_idx_str)
        state = cell["time_history"].get(str(timestep))
        if state is None:
            raise ValueError(f"No time_history for timestep {timestep} in cell {flat_idx}")

        vx_c = state["velocity"]["vx"]
        vy_c = state["velocity"]["vy"]
        vz_c = state["velocity"]["vz"]

        vx_neighbors: Dict[str, float] = {}
        vy_neighbors: Dict[str, float] = {}
        vz_neighbors: Dict[str, float] = {}

        for direction in ORDER_6:
            neighbor_idx = cell.get(f"flat_index_{direction}")
            if neighbor_idx is None:
                vx_neighbors[direction] = vx_c
                vy_neighbors[direction] = vy_c
                vz_neighbors[direction] = vz_c
                if debug and flat_idx < 5:
                    print(f"Cell {flat_idx} {cell['grid_index']} → {direction} boundary → clamped to vx={vx_c}, vy={vy_c}, vz={vz_c}")
            else:
                neighbor_state = cell_dict[str(neighbor_idx)]["time_history"].get(str(timestep))
                if neighbor_state is None:
                    raise ValueError(f"No time_history for timestep {timestep} in neighbor {neighbor_idx}")
                vx_neighbors[direction] = neighbor_state["velocity"]["vx"]
                vy_neighbors[direction] = neighbor_state["velocity"]["vy"]
                vz_neighbors[direction] = neighbor_state["velocity"]["vz"]
                if debug and flat_idx < 5:
                    print(f"Cell {flat_idx} {cell['grid_index']} → {direction} neighbor {neighbor_idx} → "
                          f"vx={vx_neighbors[direction]}, vy={vy_neighbors[direction]}, vz={vz_neighbors[direction]}")

        neighbor_map[flat_idx] = {
            "vx_neighbors": vx_neighbors,
            "vy_neighbors": vy_neighbors,
            "vz_neighbors": vz_neighbors,
        }

        if debug and flat_idx < 5:
            print(f"✅ Cell {flat_idx} neighbor map: {neighbor_map[flat_idx]}")

    if debug:
        print("✅ Neighbor map build complete.")

    return neighbor_map



