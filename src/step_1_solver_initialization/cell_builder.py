# src/step_1_solver_initialization/cell_builder.py
# 🧱 Step 1: Domain Initialization — Build the Per-Cell Dictionary

from src.step_1_solver_initialization.indexing_utils import flat_to_grid
from src.step_1_solver_initialization.neighbor_mapper import get_stencil_neighbors

debug = False

def build_cell_dict(config: dict) -> dict[int, dict]:
    """
    Build the per-cell dictionary from the simulation input config.
    Each cell entry contains:
      - flat_index
      - grid_index [i, j, k]
      - stencil-safe neighbor mapping
      - geometry classification (fluid, solid, boundary)
      - boundary role (inlet, outlet, wall, or None)
      - time_history initialized with pressure and velocity from input
    """

    # Domain shape
    nx = config["domain_definition"]["nx"]
    ny = config["domain_definition"]["ny"]
    nz = config["domain_definition"]["nz"]
    shape = (nx, ny, nz)

    # Geometry mask
    mask_flat = config["geometry_definition"]["geometry_mask_flat"]
    mask_encoding = config["geometry_definition"]["mask_encoding"]

    # Initial conditions
    init_pressure = config["initial_conditions"]["initial_pressure"]
    init_velocity = config["initial_conditions"]["initial_velocity"]

    # Boundary conditions
    boundary_conditions = config.get("boundary_conditions", [])

    cell_dict = {}

    for flat_index, mask_value in enumerate(mask_flat):
        i, j, k = flat_to_grid(flat_index, shape)

        # Neighbor mapping
        neighbors = get_stencil_neighbors(flat_index, shape)

        # Geometry classification
        if mask_value == mask_encoding["fluid"]:
            cell_type = "fluid"
        elif mask_value == mask_encoding["solid"]:
            cell_type = "solid"
        elif mask_value == mask_encoding["boundary"]:
            cell_type = "boundary"
        else:
            cell_type = "fluid"  # fallback

        # Boundary role assignment
        boundary_role = None
        if cell_type == "boundary":
            # Match boundary conditions by face
            for bc in boundary_conditions:
                apply_faces = bc.get("apply_faces", [])
                # Simplified: if any apply_faces match domain edge, assign role
                if "x_min" in apply_faces and i == 0:
                    boundary_role = bc["role"]
                elif "x_max" in apply_faces and i == nx - 1:
                    boundary_role = bc["role"]
                elif "y_min" in apply_faces and j == 0:
                    boundary_role = bc["role"]
                elif "y_max" in apply_faces and j == ny - 1:
                    boundary_role = bc["role"]
                elif "z_min" in apply_faces and k == 0:
                    boundary_role = bc["role"]
                elif "z_max" in apply_faces and k == nz - 1:
                    boundary_role = bc["role"]
                elif "wall" in apply_faces:
                    boundary_role = "wall"

        # Initialize physics history
        time_history = {
            0: {
                "pressure": init_pressure,
                "velocity": {
                    "vx": init_velocity[0],
                    "vy": init_velocity[1],
                    "vz": init_velocity[2],
                },
            }
        }

        cell_entry = {
            "flat_index": flat_index,
            "grid_index": [i, j, k],
            **neighbors,
            "cell_type": cell_type,
            "boundary_role": boundary_role,
            "time_history": time_history,
        }

        cell_dict[flat_index] = cell_entry

        if debug and flat_index < 5:
            print(f"🧱 Built cell {flat_index}: {cell_entry}")

    return cell_dict

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Build per-cell dictionary from simulation input JSON.")
    parser.add_argument("--input", required=True, help="Path to input JSON file (Step 0 output).")
    parser.add_argument("--output", required=True, help="Path to write cell_dict JSON output.")
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            config = json.load(f)

        cell_dict = build_cell_dict(config)

        with open(args.output, "w") as f:
            json.dump(cell_dict, f, indent=2)

        if debug:
            print(f"✅ Cell dictionary built and written to {args.output}")
    except Exception as e:
        print(f"❌ Error running cell_builder: {e}", file=sys.stderr)
        sys.exit(1)



