import numpy as np

def apply_boundary_conditions(velocity, pressure, boundary_conditions, mesh_info):
    """
    Enforces inlet velocity, outlet pressure, and wall (no-slip) boundary conditions.
    Assumes a structured grid for mapping face_ids to node indices.
    """
    nx, ny, nz = mesh_info["grid_shape"]
    idx_to_node = mesh_info["idx_to_node"]

    inlet_nodes_1d_indices = []
    outlet_nodes_1d_indices = []
    wall_nodes_1d_indices = set()

    # Face 1 in JSON -> x_min boundary (i=0)
    if 1 in boundary_conditions["inlet"]["faces"]:
        for j in range(ny):
            for k in range(nz):
                if (0, j, k) in idx_to_node:
                    inlet_nodes_1d_indices.append(idx_to_node[(0, j, k)])

    # Face 232 in JSON -> x_max boundary (i=nx-1)
    if 232 in boundary_conditions["outlet"]["faces"]:
        for j in range(ny):
            for k in range(nz):
                if (nx-1, j, k) in idx_to_node:
                    outlet_nodes_1d_indices.append(idx_to_node[(nx-1, j, k)])

    # Wall faces
    # Face 10 in JSON -> y_min boundary (j=0)
    if 10 in boundary_conditions["wall"]["faces"]:
        for i in range(nx):
            for k in range(nz):
                if (i, 0, k) in idx_to_node:
                    wall_nodes_1d_indices.add(idx_to_node[(i, 0, k)])
    # Face 11 in JSON -> y_max boundary (j=ny-1)
    if 11 in boundary_conditions["wall"]["faces"]:
        for i in range(nx):
            for k in range(nz):
                if (i, ny-1, k) in idx_to_node:
                    wall_nodes_1d_indices.add(idx_to_node[(i, ny-1, k)])
    
    if nz > 1:
        # Face 12 in JSON -> z_min boundary (k=0)
        if 12 in boundary_conditions["wall"]["faces"]:
            for i in range(nx):
                for j in range(ny):
                    if (i, j, 0) in idx_to_node:
                        wall_nodes_1d_indices.add(idx_to_node[(i, j, 0)])
        # Face 13 in boundary_conditions["wall"]["faces"] -> z_max boundary (k=nz-1)
        if 13 in boundary_conditions["wall"]["faces"]:
            for i in range(nx):
                for j in range(ny):
                    if (i, j, nz-1) in idx_to_node:
                        wall_nodes_1d_indices.add(idx_to_node[(i, j, nz-1)])

    inlet_vel = np.array(boundary_conditions["inlet"]["velocity"])
    for node_id in inlet_nodes_1d_indices:
        velocity[node_id] = inlet_vel
        if node_id in wall_nodes_1d_indices:
            wall_nodes_1d_indices.remove(node_id)

    if boundary_conditions["wall"]["no_slip"]:
        for node_id in wall_nodes_1d_indices:
            velocity[node_id] = np.zeros(3)
    
    outlet_pres = boundary_conditions["outlet"]["pressure"]
    for node_id in outlet_nodes_1d_indices:
        pressure[node_id] = outlet_pres