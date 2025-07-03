# src/preprocessing/identify_boundary_nodes.py

import numpy as np

TOL = 1e-6

def identify_boundary_nodes(mesh_info, mesh_faces=None):
    """
    Populates 'cell_indices' and 'ghost_indices' for each boundary condition in mesh_info.
    Infers boundary orientation from node coordinates if mesh_faces are provided.
    Supports velocity, pressure, and other physical fields.
    """
    grid_shape = mesh_info.get("grid_shape")
    if not grid_shape or len(grid_shape) != 3:
        raise ValueError("grid_shape must be a list of [nx, ny, nz]")

    nx, ny, nz = grid_shape
    nx_total, ny_total, nz_total = nx + 2, ny + 2, nz + 2

    dx = mesh_info.get("dx", 1.0)
    dy = mesh_info.get("dy", 1.0)
    dz = mesh_info.get("dz", 1.0)

    min_x = mesh_info.get("min_x", 0.0)
    min_y = mesh_info.get("min_y", 0.0)
    min_z = mesh_info.get("min_z", 0.0)

    boundary_conditions = mesh_info.get("boundary_conditions", {})
    if not boundary_conditions:
        print("[identify_boundary_nodes] Warning: No boundary conditions found to map.")
        return

    face_map = {
        "xmin": (1, slice(1, ny_total - 1), slice(1, nz_total - 1)),
        "xmax": (nx, slice(1, ny_total - 1), slice(1, nz_total - 1)),
        "ymin": (slice(1, nx_total - 1), 1, slice(1, nz_total - 1)),
        "ymax": (slice(1, nx_total - 1), ny, slice(1, nz_total - 1)),
        "zmin": (slice(1, nx_total - 1), slice(1, ny_total - 1), 1),
        "zmax": (slice(1, nx_total - 1), slice(1, ny_total - 1), nz),
    }

    ghost_cell_face_map = {
        "xmin_ghost": (0, slice(1, ny_total - 1), slice(1, nz_total - 1)),
        "xmax_ghost": (nx_total - 1, slice(1, ny_total - 1), slice(1, nz_total - 1)),
        "ymin_ghost": (slice(1, nx_total - 1), 0, slice(1, nz_total - 1)),
        "ymax_ghost": (slice(1, nx_total - 1), ny_total - 1, slice(1, nz_total - 1)),
        "zmin_ghost": (slice(1, nx_total - 1), slice(1, ny_total - 1), 0),
        "zmax_ghost": (slice(1, nx_total - 1), slice(1, ny_total - 1), nz_total - 1),
    }

    def infer_face_direction(face_nodes):
        coords = np.array(list(face_nodes.values()))
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

    face_id_direction_map = {}
    if mesh_faces:
        for face in mesh_faces:
            face_id = face["face_id"]
            direction = infer_face_direction(face["nodes"])
            if direction:
                face_id_direction_map[face_id] = direction

    for bc_name, bc_data in boundary_conditions.items():
        if "cell_indices" in bc_data and "ghost_indices" in bc_data:
            continue

        candidate_direction = None
        if "faces" in bc_data and mesh_faces:
            face_ids = bc_data["faces"]
            directions = {face_id_direction_map.get(fid) for fid in face_ids if fid in face_id_direction_map}
            directions.discard(None)
            if len(directions) == 1:
                candidate_direction = directions.pop()

        if candidate_direction is None:
            lname = bc_name.lower()
            for key in face_map:
                if key in lname:
                    candidate_direction = key
                    break

        if candidate_direction not in face_map or f"{candidate_direction}_ghost" not in ghost_cell_face_map:
            print(f"[identify_boundary_nodes] ⚠️ Could not determine geometric face for BC '{bc_name}'.")
            continue

        boundary_slice = face_map[candidate_direction]
        ghost_slice = ghost_cell_face_map[f"{candidate_direction}_ghost"]

        boundary_mask = np.zeros((nx_total, ny_total, nz_total), dtype=bool)
        ghost_mask = np.zeros((nx_total, ny_total, nz_total), dtype=bool)

        try:
            boundary_mask[boundary_slice] = True
            ghost_mask[ghost_slice] = True
        except IndexError as e:
            print(f"[identify_boundary_nodes] ⚠️ Failed to apply slice for BC '{bc_name}': {e}")
            continue

        boundary_indices = np.argwhere(boundary_mask)
        ghost_indices = np.argwhere(ghost_mask)

        bc_data["cell_indices"] = boundary_indices.tolist()
        bc_data["ghost_indices"] = ghost_indices.tolist()

        print(f"[identify_boundary_nodes] ✅ Mapped BC '{bc_name}' to face '{candidate_direction}'")
        print(f"   - Identified {len(boundary_indices)} boundary cells")
        print(f"   - Identified {len(ghost_indices)} ghost cells")

    mesh_info["boundary_conditions"] = boundary_conditions



