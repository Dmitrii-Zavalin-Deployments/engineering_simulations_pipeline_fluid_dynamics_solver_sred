# src/utils/mesh_utils.py

import sys

def get_domain_extents(boundary_faces):
    """
    Extracts the min/max x, y, z coordinates from all nodes in boundary_faces.
    Nodes are expected to be in a dictionary format: {node_id: [x,y,z]}.
    """
    all_x = []
    all_y = []
    all_z = []

    for face in boundary_faces:
        for node_coords in face["nodes"].values():
            all_x.append(node_coords[0])
            all_y.append(node_coords[1])
            all_z.append(node_coords[2])

    if not all_x or not all_y or not all_z:
        raise ValueError("No nodes found in boundary faces to determine domain extents.")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    min_z, max_z = min(all_z), max(all_z)

    return min_x, max_x, min_y, max_y, min_z, max_z


def infer_uniform_grid_parameters(min_val, max_val, all_coords_for_axis, axis_name, json_nx=None):
    """
    Infers dx and nx (or dy, ny / dz, nz) for a uniform grid along a single axis.
    Prioritizes JSON nx/ny/nz if provided and consistent with domain extent.
    """
    # Use the nx/ny/nz from JSON's domain_definition if available
    if json_nx is not None and json_nx > 0:
        inferred_dx = (max_val - min_val) / json_nx if abs(max_val - min_val) > 1e-9 else 1.0
        inferred_num_cells = json_nx
        print(f"DEBUG (infer_uniform_grid_parameters): Using JSON specified {axis_name}x={json_nx} cells. dx={inferred_dx:.6e}")
        return inferred_dx, inferred_num_cells

    # Fallback to inferring from unique boundary face coordinates if JSON nx is not provided or invalid
    unique_coords = sorted(list(set(all_coords_for_axis)))

    if not unique_coords:
        raise ValueError(f"No unique coordinates found for {axis_name}-axis in boundary faces.")

    num_unique_planes = len(unique_coords)

    if num_unique_planes == 1:
        if abs(max_val - min_val) < 1e-9:
            spacing = 1.0
            num_cells = 1
        else:
            print(f"Warning: Unexpected condition in {axis_name}-axis: 1 unique coord but max_val != min_val. Forcing {axis_name}x=1 cell and spacing=(max-min).", file=sys.stderr)
            spacing = (max_val - min_val)
            num_cells = 1
    else:
        spacing = (max_val - min_val) / (num_unique_planes - 1)
        num_cells = num_unique_planes - 1
        
        if num_cells <= 0:
            if abs(max_val - min_val) > 1e-9:
                num_cells = 1
                spacing = (max_val - min_val)
                print(f"Warning: Inferred {axis_name}x resulted in 0 or less cells despite domain extent. Forcing {axis_name}x=1 and spacing=(max-min).", file=sys.stderr)
            else:
                num_cells = 1
                spacing = 1.0
                print(f"Warning: Inferred {axis_name}x resulted in 0 or less cells for zero-extent domain. Forcing {axis_name}x=1 and nominal spacing.", file=sys.stderr)

        if spacing < 1e-9 and abs(max_val - min_val) > 1e-9:
            print(f"Warning: Extremely small calculated spacing ({spacing:.2e}) for {axis_name}-axis with significant extent. This might indicate many unique, closely clustered points.", file=sys.stderr)

    print(f"Inferred {axis_name}-axis: {num_cells} cells, spacing {spacing:.6e}")
    return spacing, num_cells