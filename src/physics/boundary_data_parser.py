# src/physics/boundary_data_parser.py

import numpy as np
import sys

# Assume a small tolerance for floating point comparisons
TOLERANCE = 1e-6

def _map_face_nodes_to_grid_indices(face_nodes_coords, mesh_info):
    """
    Maps physical coordinates of boundary face nodes to structured grid cell indices.
    """
    min_x, max_x = mesh_info['min_x'], mesh_info['max_x']
    min_y, max_y = mesh_info['min_y'], mesh_info['max_y']
    min_z, max_z = mesh_info['min_z'], mesh_info['max_z']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']
    nx, ny, nz = mesh_info['grid_shape']

    face_min_coords = np.min(face_nodes_coords, axis=0)
    face_max_coords = np.max(face_nodes_coords, axis=0)

    # Calculate min/max cell indices, clamping to grid boundaries
    # Using floor and ceil with tolerance to handle floating point precision
    i_min_face = int(np.floor((face_min_coords[0] - min_x) / dx - TOLERANCE)) if dx > TOLERANCE else 0
    i_max_face = int(np.ceil((face_max_coords[0] - min_x) / dx + TOLERANCE)) if dx > TOLERANCE else nx
    j_min_face = int(np.floor((face_min_coords[1] - min_y) / dy - TOLERANCE)) if dy > TOLERANCE else 0
    j_max_face = int(np.ceil((face_max_coords[1] - min_y) / dy + TOLERANCE)) if dy > TOLERANCE else ny
    k_min_face = int(np.floor((face_min_coords[2] - min_z) / dz - TOLERANCE)) if dz > TOLERANCE else 0
    k_max_face = int(np.ceil((face_max_coords[2] - min_z) / dz + TOLERANCE)) if dz > TOLERANCE else nz

    i_min_face = max(0, min(nx, i_min_face))
    i_max_face = max(0, min(nx, i_max_face))
    j_min_face = max(0, min(ny, j_min_face))
    j_max_face = max(0, min(ny, j_max_face))
    k_min_face = max(0, min(nz, k_min_face))
    k_max_face = max(0, min(nz, k_max_face))
    
    return i_min_face, i_max_face, j_min_face, j_max_face, k_min_face, k_max_face


def _infer_boundary_properties(overall_min_coords, overall_max_coords, mesh_info):
    """
    Infers the boundary dimension, side (min/max), and interior neighbor offset
    based on the overall extent of the boundary faces relative to the domain.
    """
    min_x, max_x = mesh_info['min_x'], mesh_info['max_x']
    min_y, max_y = mesh_info['min_y'], mesh_info['max_y']
    min_z, max_z = mesh_info['min_z'], mesh_info['max_z']

    boundary_dim = None
    boundary_side = None
    interior_neighbor_offset = np.array([0, 0, 0], dtype=int)

    # Check for alignment with domain boundaries
    if abs(overall_min_coords[0] - min_x) < TOLERANCE and abs(overall_max_coords[0] - min_x) < TOLERANCE:
        boundary_dim = 0; boundary_side = "min"; interior_neighbor_offset[0] = 1
    elif abs(overall_min_coords[0] - max_x) < TOLERANCE and abs(overall_max_coords[0] - max_x) < TOLERANCE:
        boundary_dim = 0; boundary_side = "max"; interior_neighbor_offset[0] = -1
    elif abs(overall_min_coords[1] - min_y) < TOLERANCE and abs(overall_max_coords[1] - min_y) < TOLERANCE:
        boundary_dim = 1; boundary_side = "min"; interior_neighbor_offset[1] = 1
    elif abs(overall_min_coords[1] - max_y) < TOLERANCE and abs(overall_max_coords[1] - max_y) < TOLERANCE:
        boundary_dim = 1; boundary_side = "max"; interior_neighbor_offset[1] = -1
    elif abs(overall_min_coords[2] - min_z) < TOLERANCE and abs(overall_max_coords[2] - min_z) < TOLERANCE:
        boundary_dim = 2; boundary_side = "min"; interior_neighbor_offset[2] = 1
    elif abs(overall_min_coords[2] - max_z) < TOLERANCE and abs(overall_max_coords[2] - max_z) < TOLERANCE:
        boundary_dim = 2; boundary_side = "max"; interior_neighbor_offset[2] = -1
            
    return boundary_dim, boundary_side, interior_neighbor_offset


def identify_boundary_nodes(boundary_conditions_definition, all_mesh_boundary_faces, mesh_info):
    """
    Identifies boundary conditions by mapping face_ids to structured grid cell indices.
    """
    print("DEBUG (identify_boundary_nodes): Starting boundary node identification.")
    
    mesh_face_lookup = {face['face_id']: face for face in all_mesh_boundary_faces}
    print(f"DEBUG (identify_boundary_nodes): Mesh face lookup created with {len(mesh_face_lookup)} entries.")

    processed_bcs = {}
    nx, ny, nz = mesh_info['grid_shape']

    for bc_name, bc_properties in boundary_conditions_definition.items():
        print(f"DEBUG (identify_boundary_nodes): Processing BC '{bc_name}'.")
        bc_type = bc_properties.get("type", bc_name) # Default type to name if not specified

        bc_faces_ids = bc_properties.get("faces", [])
        if not bc_faces_ids:
            print(f"Warning: Boundary condition '{bc_name}' has no 'faces' specified. Skipping.", file=sys.stderr)
            continue
        print(f"DEBUG (identify_boundary_nodes): BC '{bc_name}' has {len(bc_faces_ids)} associated faces.")

        current_bc_cell_indices = set()
        overall_min_coords_boundary = np.array([np.inf, np.inf, np.inf])
        overall_max_coords_boundary = np.array([-np.inf, -np.inf, -np.inf])

        for face_id in bc_faces_ids:
            mesh_face_data = mesh_face_lookup.get(face_id)
            if not mesh_face_data:
                print(f"Warning: Mesh face with face_id {face_id} not found for boundary '{bc_name}'. Skipping.", file=sys.stderr)
                continue
            
            face_nodes_coords = np.array(list(mesh_face_data['nodes'].values()), dtype=np.float64)
            
            # Update overall boundary extents
            overall_min_coords_boundary = np.minimum(overall_min_coords_boundary, np.min(face_nodes_coords, axis=0))
            overall_max_coords_boundary = np.maximum(overall_max_coords_boundary, np.max(face_nodes_coords, axis=0))

            # Map face nodes to grid cell indices
            i_min, i_max, j_min, j_max, k_min, k_max = _map_face_nodes_to_grid_indices(face_nodes_coords, mesh_info)
            print(f"DEBUG (identify_boundary_nodes): Face {face_id} for '{bc_name}' maps to i[{i_min}:{i_max}], j[{j_min}:{j_max}], k[{k_min}:{k_max}]")

            # Add all cells within the inferred range to the set
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        current_bc_cell_indices.add((i, j, k))
        
        if not current_bc_cell_indices:
            print(f"Warning: No structured grid cells found for boundary '{bc_name}'. Skipping application.", file=sys.stderr)
            continue
        
        final_cell_indices_array = np.array(list(current_bc_cell_indices), dtype=int)
        print(f"DEBUG (identify_boundary_nodes): BC '{bc_name}' identified {final_cell_indices_array.shape[0]} unique cells.")

        # Infer boundary properties (dimension, side, offset)
        boundary_dim, boundary_side, interior_neighbor_offset = \
            _infer_boundary_properties(overall_min_coords_boundary, overall_max_coords_boundary, mesh_info)
        
        if boundary_dim is None:
            print(f"Warning: Boundary '{bc_name}' does not perfectly align with an axis-aligned plane. "
                  "Neumann/Pressure Outlet conditions may behave unexpectedly.", file=sys.stderr)

        apply_to = bc_properties.get("apply_to", ["velocity", "pressure"])
        if bc_name == "wall" and bc_properties.get("no_slip", False):
            apply_to = ["velocity"]

        processed_bcs[bc_name] = {
            "type": bc_type,
            "cell_indices": final_cell_indices_array,
            "velocity": np.array(bc_properties.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
            "pressure": bc_properties.get("pressure", 0.0),
            "apply_to": apply_to,
            "boundary_dim": boundary_dim,
            "boundary_side": boundary_side,
            "interior_neighbor_offset": interior_neighbor_offset
        }
    
    print("DEBUG (identify_boundary_nodes): Finished boundary node identification. Returning processed_bcs.")
    return processed_bcs