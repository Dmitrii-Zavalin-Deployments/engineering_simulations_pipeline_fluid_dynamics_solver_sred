# src/physics/pressure_projection.py
# 🔁 Pressure Projection — solves ∇²P = ∇ · u and applies velocity correction for incompressibility
# 📌 This module enforces incompressibility via pressure solve and velocity projection.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.physics.pressure_methods.jacobi import solve_jacobi_pressure
from src.physics.pressure_methods.utils import index_fluid_cells
from src.physics.velocity_projection import apply_pressure_velocity_projection
from src.utils.ghost_registry import build_ghost_registry, extract_ghost_coordinates

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def solve_pressure_poisson(
    grid: List[Cell],
    divergence: List[float],
    config: dict,
    verbose: bool = False
) -> Tuple[List[Cell], bool, Dict]:
    """
    Computes updated pressure values for fluid cells using the selected solver method,
    then projects velocity to enforce incompressibility.

    Returns:
        Tuple[List[Cell], bool, Dict]: 
            - Grid with updated pressure and velocity values
            - pressure_mutated flag indicating if any fluid pressure changed
            - ghost_registry for downstream diagnostics
    """
    method = config.get("pressure_solver", {}).get("method", "jacobi").lower()

    # 🔍 Index fluid cells for solver
    fluid_coords = index_fluid_cells(grid)
    fluid_cell_count = len(fluid_coords)

    if len(divergence) != fluid_cell_count:
        raise ValueError(
            f"Divergence list length ({len(divergence)}) does not match number of fluid cells ({fluid_cell_count})"
        )

    if debug or verbose:
        for i, d in enumerate(divergence):
            print(f"[PRESSURE] Divergence[{i}] = {d:.6e}")

    # 🧱 Prepare ghost registry
    ghost_registry = build_ghost_registry(grid)
    ghost_coords = extract_ghost_coordinates(ghost_registry)
    ghost_set = set(ghost_coords)

    spacing = (
        (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"],
        (config["domain_definition"]["max_y"] - config["domain_definition"]["min_y"]) / config["domain_definition"]["ny"],
        (config["domain_definition"]["max_z"] - config["domain_definition"]["min_z"]) / config["domain_definition"]["nz"]
    )

    # 🔁 Select solver method
    if method == "jacobi":
        pressure_values = solve_jacobi_pressure(grid, divergence, config, ghost_coords)
    else:
        raise ValueError(f"Unknown or unsupported pressure solver method: '{method}'")

    # 🧱 Reconstruct grid and track pressure mutation
    updated = []
    fluid_index = 0
    pressure_mutated = False

    for cell in grid:
        coord = (cell.x, cell.y, cell.z)

        if cell.fluid_mask:
            new_pressure = pressure_values[fluid_index]
            old_pressure = cell.pressure if isinstance(cell.pressure, float) else 0.0
            delta = abs(old_pressure - new_pressure)

            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=cell.velocity[:] if isinstance(cell.velocity, list) else None,
                pressure=new_pressure,
                fluid_mask=True
            )

            # ✅ Reflex mutation tagging
            if delta > 1e-6:
                pressure_mutated = True
                updated_cell.pressure_mutated = True
                updated_cell.mutation_source = "pressure_solver"
                updated_cell.mutation_step = config.get("step_index", None)
                updated_cell.pressure_delta = round(delta, 6)
                if debug or verbose:
                    print(f"[MUTATION] Pressure changed @ {coord} → ΔP = {delta:.2e}")

            # ✅ Reflex traceability: ghost adjacency tagging
            for ghost in ghost_set:
                if all(abs(a - b) <= spacing[i] + 1e-3 for i, (a, b) in enumerate(zip(coord, ghost))):
                    updated_cell.influenced_by_ghost = True
                    updated_cell.mutation_triggered_by = "ghost_influence"
                    if debug or verbose:
                        print(f"[TRACE] Ghost-influenced mutation @ {coord}")
                    break

            fluid_index += 1
        else:
            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=None,
                pressure=None,
                fluid_mask=False
            )

        updated.append(updated_cell)

    # 💨 Apply pressure-based velocity projection: u ← u - ∇P
    projected_grid = apply_pressure_velocity_projection(updated, config)

    return projected_grid, pressure_mutated, ghost_registry



