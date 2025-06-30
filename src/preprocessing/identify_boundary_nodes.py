# src/preprocessing/identify_boundary_nodes.py

import numpy as np

TOL = 1e-6

def identify_boundary_nodes(mesh_info, mesh_faces=None):
    """
    Populates 'cell_indices' for each boundary condition in the mesh_info dict.
    Automatically infers boundary orientation from node coordinates if 'faces' are provided.
    This function is agnostic to the type of boundary condition (e.g., velocity or pressure)
    and focuses solely on mapping the geometric faces to grid cell indices.

    Args:
        mesh_info (dict): Must include 'grid_shape' and 'boundary_conditions'.
        mesh_faces (list): Optional list of 'boundary_faces' (from input_data['mesh']).
    """
    grid_shape = mesh_info.get("grid_shape")
    if not grid_shape or len(grid_shape) != 3:
        raise ValueError("grid_shape must be a list of [nx, ny, nz]")

    # The grid dimensions include the core domain and ghost cells (nx+2, ny+2, nz+2)
    # The boundary cells are on the edges of the core domain.
    nx, ny, nz = grid_shape
    nx_total = nx + 2  # Total size including ghost cells
    ny_total = ny + 2
    nz_total = nz + 2

    dx = mesh_info.get("dx", 1.0)
    dy = mesh_info.get("dy", 1.0)
    dz = mesh_info.get("dz", 1.0)

    min_x = mesh_info.get("min_x", 0.0)
    min_y = mesh_info.get("min_y", 0.0)
    min_z = mesh_info.get("min_z", 0.0)

    # The boundary conditions are now in a nested dict 'data'
    boundary_conditions = mesh_info.get("boundary_conditions", {})
    if not boundary_conditions:
        # It's not an error if there are no BCs, just a warning.
        print("[identify_boundary_nodes] Warning: No boundary conditions found to map.")
        return

    # Define slices for the interior domain faces (which correspond to boundary cells)
    face_map = {
        # These are the indices for the first layer of *core* domain cells.
        # Ghost cells are at index 0 and nx_total-1, etc.
        "xmin": (1, slice(1, ny_total - 1), slice(1, nz_total - 1)),
        "xmax": (nx, slice(1, ny_total - 1), slice(1, nz_total - 1)), # index 'nx' is the last core cell before ghost
        "ymin": (slice(1, nx_total - 1), 1, slice(1, nz_total - 1)),
        "ymax": (slice(1, nx_total - 1), ny, slice(1, nz_total - 1)),
        "zmin": (slice(1, nx_total - 1), slice(1, ny_total - 1), 1),
        "zmax": (slice(1, nx_total - 1), slice(1, ny_total - 1), nz),
    }
    
    # Define slices for the ghost cells. We will apply BCs to these.
    ghost_cell_face_map = {
        "xmin_ghost": (0, slice(1, ny_total - 1), slice(1, nz_total - 1)),
        "xmax_ghost": (nx_total - 1, slice(1, ny_total - 1), slice(1, nz_total - 1)),
        "ymin_ghost": (slice(1, nx_total - 1), 0, slice(1, nz_total - 1)),
        "ymax_ghost": (slice(1, nx_total - 1), ny_total - 1, slice(1, nz_total - 1)),
        "zmin_ghost": (slice(1, nx_total - 1), slice(1, ny_total - 1), 0),
        "zmax_ghost": (slice(1, nx_total - 1), slice(1, ny_total - 1), nz_total - 1),
    }
    
    def infer_face_direction(face_nodes):
        """Infers the geometric face from its node coordinates."""
        coords = np.array(list(face_nodes.values()))
        # Check against the physical domain extents
        is_xmin = np.all(np.abs(coords[:, 0] - min_x) < TOL)
        is_xmax = np.all(np.abs(coords[:, 0] - (min_x + dx * nx)) < TOL)
        is_ymin = np.all(np.abs(coords[:, 1] - min_y) < TOL)
        is_ymax = np.all(np.abs(coords[:, 1] - (min_y + dy * ny)) < TOL)
        is_zmin = np.all(np.abs(coords[:, 2] - min_z) < TOL)
        is_zmax = np.all(np.abs(coords[:, 2] - (min_z + dz * nz)) < TOL)

        if is_xmin: return "xmin"
        if is_xmax: return "xmax"
        if is_ymin: return "ymin"
        if is_ymax: return "ymax"
        if is_zmin: return "zmin"
        if is_zmax: return "zmax"
        return None

    # Build a mapping of face_id -> direction for faster lookup
    face_id_direction_map = {}
    if mesh_faces:
        for face in mesh_faces:
            face_id = face["face_id"]
            direction = infer_face_direction(face["nodes"])
            if direction:
                face_id_direction_map[face_id] = direction

    # Iterate through each defined boundary condition to map it to grid indices
    for bc_name, bc_data in boundary_conditions.items():
        if "cell_indices" in bc_data and "ghost_indices" in bc_data:
            # Skip if already processed
            continue

        # Try to detect direction via face reference from the input mesh
        candidate_direction = None
        if "faces" in bc_data and mesh_faces:
            face_ids = bc_data["faces"]
            directions = set()
            for fid in face_ids:
                dir = face_id_direction_map.get(fid)
                if dir:
                    directions.add(dir)
            if len(directions) == 1:
                candidate_direction = list(directions)[0]

        # Fall back to matching on the boundary condition label (e.g., "inlet_xmin")
        if candidate_direction is None:
            lower_name = bc_name.lower()
            if "xmin" in lower_name: candidate_direction = "xmin"
            elif "xmax" in lower_name: candidate_direction = "xmax"
            elif "ymin" in lower_name: candidate_direction = "ymin"
            elif "ymax" in lower_name: candidate_direction = "ymax"
            elif "zmin" in lower_name: candidate_direction = "zmin"
            elif "zmax" in lower_name: candidate_direction = "zmax"

        if candidate_direction not in face_map:
            print(f"[identify_boundary_nodes] ⚠️ Could not determine geometric face for BC '{bc_name}'.")
            continue

        # Get the slicing for the boundary cells and ghost cells
        boundary_slice = face_map[candidate_direction]
        ghost_slice = ghost_cell_face_map[f"{candidate_direction}_ghost"]

        # Create masks and find indices
        boundary_mask = np.full((nx_total, ny_total, nz_total), False)
        boundary_mask[boundary_slice] = True
        boundary_indices = np.argwhere(boundary_mask)

        ghost_mask = np.full((nx_total, ny_total, nz_total), False)
        ghost_mask[ghost_slice] = True
        ghost_indices = np.argwhere(ghost_mask)

        # Store the identified indices in the mesh_info dictionary
        # We convert to list for JSON serialization
        bc_data["cell_indices"] = boundary_indices.tolist()
        bc_data["ghost_indices"] = ghost_indices.tolist()

        print(f"[identify_boundary_nodes] ✅ Mapped BC '{bc_name}' to face '{candidate_direction}'.")
        print(f"   - Identified {len(boundary_indices)} boundary cells.")
        print(f"   - Identified {len(ghost_indices)} ghost cells.")

    mesh_info["boundary_conditions"] = boundary_conditions


