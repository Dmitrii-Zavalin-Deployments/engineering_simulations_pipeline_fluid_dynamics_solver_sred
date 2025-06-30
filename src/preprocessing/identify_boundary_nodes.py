# src/preprocessing/identify_boundary_nodes.py

import numpy as np

def identify_boundary_nodes(mesh_info):
    """
    Populates 'cell_indices' for each boundary condition in the mesh_info dict.
    Works for structured 3D Cartesian grids with ghost cell padding.

    Args:
        mesh_info (dict): Must include 'grid_shape' and 'boundary_conditions'
                          Modifies mesh_info['boundary_conditions'][...]['cell_indices']
    """
    grid_shape = mesh_info.get("grid_shape")
    if not grid_shape or len(grid_shape) != 3:
        raise ValueError("grid_shape must be a list of [nx, ny, nz]")

    nx, ny, nz = grid_shape
    nx += 2  # ghost padding
    ny += 2
    nz += 2

    boundary_conditions = mesh_info.get("boundary_conditions", {})
    if not boundary_conditions:
        raise ValueError("No boundary_conditions found in mesh_info")

    def match_face(bc_name):
        """
        Maps common names to grid face selection logic.
        """
        face_map = {
            "xmin": lambda: (1, slice(1, ny-1), slice(1, nz-1)),
            "xmax": lambda: (nx-2, slice(1, ny-1), slice(1, nz-1)),
            "ymin": lambda: (slice(1, nx-1), 1, slice(1, nz-1)),
            "ymax": lambda: (slice(1, nx-1), ny-2, slice(1, nz-1)),
            "zmin": lambda: (slice(1, nx-1), slice(1, ny-1), 1),
            "zmax": lambda: (slice(1, nx-1), slice(1, ny-1), nz-2),
        }
        return face_map.get(bc_name.lower())

    for bc_name, bc in boundary_conditions.items():
        if "cell_indices" in bc:
            continue  # already populated

        selector = match_face(bc_name)
        if selector is None:
            print(f"[identify_boundary_nodes] ⚠️ Unknown BC face '{bc_name}' — skipping")
            continue

        index_array = np.argwhere(np.full((nx, ny, nz), True))
        mask = np.full((nx, ny, nz), False)
        mask[selector()] = True
        indices = np.argwhere(mask)

        if indices.size == 0:
            print(f"[identify_boundary_nodes] ⚠️ No indices found for '{bc_name}'")
        else:
            print(f"[identify_boundary_nodes] ✅ Found {len(indices)} cells for '{bc_name}'")

        bc["cell_indices"] = indices.tolist()

    # Update back into mesh_info
    mesh_info["boundary_conditions"] = boundary_conditions



