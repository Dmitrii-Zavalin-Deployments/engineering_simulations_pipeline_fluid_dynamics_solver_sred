# src/preprocessing/identify_boundary_nodes.py

import numpy as np

TOL = 1e-6

def identify_boundary_nodes(mesh_info, mesh_faces=None):
    """
    Populates 'cell_indices' for each boundary condition in the mesh_info dict.
    Automatically infers boundary orientation from node coordinates if 'faces' are provided.

    Args:
        mesh_info (dict): Must include 'grid_shape' and 'boundary_conditions'.
        mesh_faces (list): Optional list of 'boundary_faces' (from input_data['mesh']).
    """
    grid_shape = mesh_info.get("grid_shape")
    if not grid_shape or len(grid_shape) != 3:
        raise ValueError("grid_shape must be a list of [nx, ny, nz]")

    nx, ny, nz = grid_shape
    nx += 2  # ghost cells
    ny += 2
    nz += 2

    dx = mesh_info.get("dx", 1.0)
    dy = mesh_info.get("dy", 1.0)
    dz = mesh_info.get("dz", 1.0)

    min_x = mesh_info.get("min_x", 0.0)
    min_y = mesh_info.get("min_y", 0.0)
    min_z = mesh_info.get("min_z", 0.0)

    boundary_conditions = mesh_info.get("boundary_conditions", {})
    if not boundary_conditions:
        raise ValueError("No boundary_conditions found in mesh_info")

    face_map = {
        "xmin": (1, slice(1, ny - 1), slice(1, nz - 1)),
        "xmax": (nx - 2, slice(1, ny - 1), slice(1, nz - 1)),
        "ymin": (slice(1, nx - 1), 1, slice(1, nz - 1)),
        "ymax": (slice(1, nx - 1), ny - 2, slice(1, nz - 1)),
        "zmin": (slice(1, nx - 1), slice(1, ny - 1), 1),
        "zmax": (slice(1, nx - 1), slice(1, ny - 1), nz - 2),
    }

    def infer_face_direction(face_nodes):
        coords = np.array(list(face_nodes.values()))
        is_xmin = np.all(np.abs(coords[:, 0] - min_x) < TOL)
        is_xmax = np.all(np.abs(coords[:, 0] - (min_x + dx * (nx - 2))) < TOL)
        is_ymin = np.all(np.abs(coords[:, 1] - min_y) < TOL)
        is_ymax = np.all(np.abs(coords[:, 1] - (min_y + dy * (ny - 2))) < TOL)
        is_zmin = np.all(np.abs(coords[:, 2] - min_z) < TOL)
        is_zmax = np.all(np.abs(coords[:, 2] - (min_z + dz * (nz - 2))) < TOL)

        if is_xmin:
            return "xmin"
        elif is_xmax:
            return "xmax"
        elif is_ymin:
            return "ymin"
        elif is_ymax:
            return "ymax"
        elif is_zmin:
            return "zmin"
        elif is_zmax:
            return "zmax"
        return None

    # Build a mapping of face_id -> direction
    face_id_direction_map = {}
    if mesh_faces:
        for face in mesh_faces:
            face_id = face["face_id"]
            direction = infer_face_direction(face["nodes"])
            if direction:
                face_id_direction_map[face_id] = direction

    for bc_name, bc in boundary_conditions.items():
        if "cell_indices" in bc:
            continue

        # Try to detect direction via face reference
        candidate_direction = None
        if "faces" in bc and mesh_faces:
            face_ids = bc["faces"]
            directions = set()
            for fid in face_ids:
                dir = face_id_direction_map.get(fid)
                if dir:
                    directions.add(dir)
            if len(directions) == 1:
                candidate_direction = list(directions)[0]

        # Fall back to matching on name
        if candidate_direction is None:
            lower_name = bc_name.lower()
            if lower_name in face_map:
                candidate_direction = lower_name

        if candidate_direction not in face_map:
            print(f"[identify_boundary_nodes] ⚠️ Could not determine face direction for '{bc_name}'")
            continue

        # Apply face mask
        mask = np.full((nx, ny, nz), False)
        mask[face_map[candidate_direction]] = True
        indices = np.argwhere(mask)

        bc["cell_indices"] = indices.tolist()
        print(f"[identify_boundary_nodes] ✅ {bc_name} → {candidate_direction}, {len(indices)} cells")

    mesh_info["boundary_conditions"] = boundary_conditions



