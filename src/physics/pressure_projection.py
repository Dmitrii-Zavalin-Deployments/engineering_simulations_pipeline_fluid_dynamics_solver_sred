# src/physics/pressure_projection.py
# 🔁 Pressure Projection — solves ∇²P = ∇ · u and applies velocity correction for incompressibility

from typing import List, Tuple, Set
from src.grid_modules.cell import Cell
from src.physics.pressure_methods.jacobi import solve_jacobi_pressure
from src.physics.pressure_methods.utils import index_fluid_cells
from src.physics.velocity_projection import apply_pressure_velocity_projection

def extract_ghost_coords(grid: List[Cell]) -> Set[Tuple[float, float, float]]:
    """
    Extract coordinates of ghost cells in the grid.

    Roadmap Alignment:
    Boundary Enforcement:
    - Ghost cells excluded from pressure solve
    - Preserves Dirichlet/Neumann enforcement logic

    Purpose:
    - Prevent ghost contamination in ∇²P solve
    - Support modular boundary tagging and reflex traceability

    Args:
        grid (List[Cell]): Full grid including fluid and ghost cells

    Returns:
        Set[Tuple]: Coordinates of ghost cells
    """
    return {
        (cell.x, cell.y, cell.z)
        for cell in grid
        if not cell.fluid_mask and hasattr(cell, "ghost_face")
    }

def solve_pressure_poisson(grid: List[Cell], divergence: List[float], config: dict) -> Tuple[List[Cell], bool]:
    """
    Computes updated pressure values for fluid cells using the selected solver method,
    then projects velocity to enforce incompressibility.

    Roadmap Alignment:
    Governing Equation:
        Continuity: ∇ · u = 0
        Pressure Solve: ∇²P = ∇ · u
        Velocity Correction: u ← u - ∇P

    Modular Enforcement:
    - Fluid indexing → pressure_methods.utils
    - Ghost exclusion → extract_ghost_coords
    - Pressure solve → jacobi.py
    - Velocity projection → velocity_projection.py

    Purpose:
    - Enforce incompressibility by solving pressure Poisson equation
    - Couple divergence diagnostics to pressure mutation
    - Apply velocity projection to complete continuity enforcement

    Returns:
        Tuple[List[Cell], bool]: 
            - Grid with updated pressure and velocity values
            - pressure_mutated flag indicating if any fluid pressure changed
    """
    method = config.get("pressure_solver", {}).get("method", "jacobi").lower()

    # 🔍 Index fluid cells for solver
    fluid_coords = index_fluid_cells(grid)
    fluid_cell_count = len(fluid_coords)

    if len(divergence) != fluid_cell_count:
        raise ValueError(
            f"Divergence list length ({len(divergence)}) does not match number of fluid cells ({fluid_cell_count})"
        )

    # 🧱 Prepare ghost info
    ghost_coords = extract_ghost_coords(grid)
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
            delta = abs(cell.pressure - new_pressure) if isinstance(cell.pressure, float) else 0.0

            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=cell.velocity[:] if isinstance(cell.velocity, list) else None,
                pressure=new_pressure,
                fluid_mask=True
            )

            if delta > 1e-6:
                pressure_mutated = True
                updated_cell.pressure_mutated = True
                updated_cell.mutation_source = "pressure_solver"
                updated_cell.mutation_step = config.get("step_index", None)

            # Reflex traceability: ghost adjacency tagging
            for ghost in ghost_set:
                if all(abs(a - b) <= spacing[i] + 1e-3 for i, (a, b) in enumerate(zip(coord, ghost))):
                    updated_cell.influenced_by_ghost = True
                    updated_cell.mutation_triggered_by = "ghost_influence"
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

    return projected_grid, pressure_mutated



