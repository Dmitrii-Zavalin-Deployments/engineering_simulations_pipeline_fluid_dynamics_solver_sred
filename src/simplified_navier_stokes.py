import os
import json
import numpy as np
import math

# --- Helper functions for synthetic grid and operations ---

def create_structured_grid_info(grid_dims, domain_size=(1.0, 1.0, 1.0)):
    """
    Creates synthetic 3D structured grid information based on explicit grid dimensions.
    Returns:
        num_nodes (int): Total number of nodes (Nx * Ny * Nz).
        nodes_coords (np.array): (num_nodes, 3) array of node coordinates.
        grid_shape (tuple): (Nx, Ny, Nz) dimensions of the grid.
        dx, dy, dz (float): Grid spacing in each dimension.
        node_to_idx (dict): Maps 1D node index to 3D (i,j,k) grid indices.
        idx_to_node (dict): Maps 3D (i,j,k) grid indices to 1D node index.
    """
    nx, ny, nz = grid_dims
    num_nodes = nx * ny * nz
    
    # Calculate grid spacing. Ensure dx, dy, dz are not zero for 1D/2D cases
    dx = domain_size[0] / (nx - 1) if nx > 1 else domain_size[0]
    dy = domain_size[1] / (ny - 1) if ny > 1 else domain_size[1]
    dz = domain_size[2] / (nz - 1) if nz > 1 else domain_size[2] # Use domain_size[2] even if nz=1
    if nz == 1: # For 2D cases, if dz is 0, set it to dx or dy for calculations involving it
        dz = dx # A pragmatic choice, though ideally dz is irrelevant if nz=1

    nodes_coords = np.zeros((num_nodes, 3))
    node_to_idx = {}
    idx_to_node = {}
    
    count = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes_coords[count, 0] = i * dx
                nodes_coords[count, 1] = j * dy
                nodes_coords[count, 2] = k * dz
                node_to_idx[count] = (i, j, k)
                idx_to_node[(i, j, k)] = count
                count += 1
    
    return num_nodes, nodes_coords, grid_dims, dx, dy, dz, node_to_idx, idx_to_node

def find_optimal_grid_dimensions(num_nodes):
    """
    Finds three integer factors (nx, ny, nz) for num_nodes such that they are as close
    to each other as possible, prioritizing nz >= 1.
    """
    factors = []
    # Find all factors of num_nodes
    for i in range(1, int(math.sqrt(num_nodes)) + 1):
        if num_nodes % i == 0:
            factors.append(i)
            if i * i != num_nodes:
                factors.append(num_nodes // i)
    factors.sort()

    best_dims = (1, 1, num_nodes) # Default to a 1D line if nothing better found
    min_diff = float('inf')

    # Try to find three factors that are close to the cube root
    cube_root = round(num_nodes**(1/3))
    
    # Iterate through possible combinations to find the "most cubic"
    # This is a heuristic and might not be perfectly optimal for all numbers
    for nz_val in factors:
        if num_nodes % nz_val == 0:
            remaining_nodes = num_nodes // nz_val
            for nx_val in factors:
                if remaining_nodes % nx_val == 0:
                    ny_val = remaining_nodes // nx_val
                    if nx_val * ny_val * nz_val == num_nodes:
                        # Calculate a "balance" metric, e.g., sum of absolute differences from cube_root
                        diff = abs(nx_val - cube_root) + abs(ny_val - cube_root) + abs(nz_val - cube_root)
                        if diff < min_diff:
                            min_diff = diff
                            best_dims = tuple(sorted((nx_val, ny_val, nz_val)))
    
    # If num_nodes is prime or has few factors, this might still return elongated dimensions.
    # For 624 nodes, (8, 13, 6) is a good factorization, which this logic should find.
    # We ensure nx, ny, nz are at least 1.
    return best_dims if best_dims[0] >= 1 and best_dims[1] >= 1 and best_dims[2] >= 1 else (1,1,num_nodes)

# --- Core Solver Functions ---

def load_json(filename):
    """
    Loads fluid simulation data from JSON file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Error: Input file not found at {filename}")
    with open(filename, "r") as file:
        return json.load(file)

def initialize_fields(num_nodes, initial_velocity, initial_pressure):
    """
    Initializes velocity and pressure fields for the entire domain.
    Assumes a uniform initial state based on inlet conditions.
    """
    velocity = np.full((num_nodes, 3), initial_velocity) # Initialize velocity to inlet velocity
    pressure = np.full(num_nodes, initial_pressure)     # Initialize pressure to inlet pressure
    return velocity, pressure

def apply_boundary_conditions(velocity, pressure, boundary_conditions, mesh_info):
    """
    Enforces inlet velocity, outlet pressure, and wall (no-slip) boundary conditions.
    Assumes a structured grid for mapping face_ids to node indices.
    """
    nx, ny, nz = mesh_info["grid_shape"]
    idx_to_node = mesh_info["idx_to_node"]

    inlet_nodes_1d_indices = []
    outlet_nodes_1d_indices = []
    wall_nodes_1d_indices = set() # Use a set to avoid duplicates

    # Example mapping: Assuming 'faces' values in JSON refer to logical grid boundaries
    # Face 1 in JSON -> x_min boundary (i=0)
    if 1 in boundary_conditions["inlet"]["faces"]:
        for j in range(ny):
            for k in range(nz):
                inlet_nodes_1d_indices.append(idx_to_node[(0, j, k)])

    # Face 232 in JSON -> x_max boundary (i=nx-1)
    if 232 in boundary_conditions["outlet"]["faces"]:
        for j in range(ny):
            for k in range(nz):
                outlet_nodes_1d_indices.append(idx_to_node[(nx-1, j, k)])

    # Wall faces: Add all boundary nodes to a set for no-slip
    # Face 10 in JSON -> y_min boundary (j=0)
    if 10 in boundary_conditions["wall"]["faces"]:
        for i in range(nx):
            for k in range(nz):
                wall_nodes_1d_indices.add(idx_to_node[(i, 0, k)])
    # Face 11 in JSON -> y_max boundary (j=ny-1)
    if 11 in boundary_conditions["wall"]["faces"]:
        for i in range(nx):
            for k in range(nz):
                wall_nodes_1d_indices.add(idx_to_node[(i, ny-1, k)])
    
    if nz > 1: # Only if 3D (if nz is effectively > 1)
        # Face 12 in JSON -> z_min boundary (k=0)
        if 12 in boundary_conditions["wall"]["faces"]:
            for i in range(nx):
                for j in range(ny):
                    wall_nodes_1d_indices.add(idx_to_node[(i, j, 0)])
        # Face 13 in boundary_conditions["wall"]["faces"] -> z_max boundary (k=nz-1)
        if 13 in boundary_conditions["wall"]["faces"]:
            for i in range(nx):
                for j in range(ny):
                    wall_nodes_1d_indices.add(idx_to_node[(i, j, nz-1)])

    # Apply Inlet Velocity (Dirichlet BC) - Note: inlet overrides wall if same node
    inlet_vel = np.array(boundary_conditions["inlet"]["velocity"])
    for node_id in inlet_nodes_1d_indices:
        velocity[node_id] = inlet_vel
        # Remove from wall nodes if also an inlet node, to ensure inlet BC takes precedence
        if node_id in wall_nodes_1d_indices:
            wall_nodes_1d_indices.remove(node_id)

    # Apply Wall (No-Slip) Boundary Conditions (Dirichlet BC for velocity)
    if boundary_conditions["wall"]["no_slip"]:
        for node_id in wall_nodes_1d_indices:
            velocity[node_id] = np.zeros(3) # Set all velocity components to zero
    
    # Apply Outlet Pressure (Dirichlet BC)
    outlet_pres = boundary_conditions["outlet"]["pressure"]
    for node_id in outlet_nodes_1d_indices:
        pressure[node_id] = outlet_pres


def compute_next_step(velocity, pressure, mesh_info, fluid_properties, dt):
    """
    Computes velocity and pressure updates using explicit Euler method
    and simplified finite difference approximations for Navier-Stokes terms.
    Applies a basic pressure projection method for incompressibility.
    Uses first-order upwind scheme for advection for improved stability.
    """
    viscosity = fluid_properties["viscosity"]
    density = fluid_properties["density"]
    
    num_nodes = mesh_info["nodes"]
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    idx_to_node = mesh_info["idx_to_node"]
    node_to_idx = mesh_info["node_to_idx"]

    u_tentative = np.copy(velocity)
    
    advection_accel = np.zeros_like(velocity)
    diffusion_accel = np.zeros_like(velocity)
    
    for node_1d_idx in range(num_nodes):
        i, j, k = node_to_idx[node_1d_idx]

        u_curr, v_curr, w_curr = velocity[node_1d_idx]

        # --- Advection Term (u . grad(u)) using First-Order Upwind ---
        # Initialize to 0, if derivative cannot be computed
        du_dx, du_dy, du_dz = 0.0, 0.0, 0.0
        dv_dx, dv_dy, dv_dz = 0.0, 0.0, 0.0
        dw_dx, dw_dy, dw_dz = 0.0, 0.0, 0.0

        # d/dx terms (for u, v, w components)
        # Upwind based on u_curr
        if u_curr >= 0: # Use backward difference
            if i > 0:
                du_dx = (u_curr - velocity[idx_to_node[(i-1, j, k)], 0]) / dx
                dv_dx = (v_curr - velocity[idx_to_node[(i-1, j, k)], 1]) / dx
                dw_dx = (w_curr - velocity[idx_to_node[(i-1, j, k)], 2]) / dx
            elif nx == 1: # 1D in x, no spatial derivative
                du_dx, dv_dx, dw_dx = 0.0, 0.0, 0.0
        else: # u_curr < 0, use forward difference
            if i < nx - 1:
                du_dx = (velocity[idx_to_node[(i+1, j, k)], 0] - u_curr) / dx
                dv_dx = (velocity[idx_to_node[(i+1, j, k)], 1] - v_curr) / dx
                dw_dx = (velocity[idx_to_node[(i+1, j, k)], 2] - w_curr) / dx
            elif nx == 1:
                du_dx, dv_dx, dw_dx = 0.0, 0.0, 0.0

        # d/dy terms (for u, v, w components)
        # Upwind based on v_curr
        if v_curr >= 0: # Use backward difference
            if j > 0:
                du_dy = (u_curr - velocity[idx_to_node[(i, j-1, k)], 0]) / dy
                dv_dy = (v_curr - velocity[idx_to_node[(i, j-1, k)], 1]) / dy
                dw_dy = (w_curr - velocity[idx_to_node[(i, j-1, k)], 2]) / dy
            elif ny == 1:
                du_dy, dv_dy, dw_dy = 0.0, 0.0, 0.0
        else: # v_curr < 0, use forward difference
            if j < ny - 1:
                du_dy = (velocity[idx_to_node[(i, j+1, k)], 0] - u_curr) / dy
                dv_dy = (velocity[idx_to_node[(i, j+1, k)], 1] - v_curr) / dy
                dw_dy = (velocity[idx_to_node[(i, j+1, k)], 2] - w_curr) / dy
            elif ny == 1:
                du_dy, dv_dy, dw_dy = 0.0, 0.0, 0.0

        # d/dz terms (for u, v, w components)
        # Upwind based on w_curr (only if nz > 1)
        if nz > 1:
            if w_curr >= 0: # Use backward difference
                if k > 0:
                    du_dz = (u_curr - velocity[idx_to_node[(i, j, k-1)], 0]) / dz
                    dv_dz = (v_curr - velocity[idx_to_node[(i, j, k-1)], 1]) / dz
                    dw_dz = (w_curr - velocity[idx_to_node[(i, j, k-1)], 2]) / dz
            else: # w_curr < 0, use forward difference
                if k < nz - 1:
                    du_dz = (velocity[idx_to_node[(i, j, k+1)], 0] - u_curr) / dz
                    dv_dz = (velocity[idx_to_node[(i, j, k+1)], 1] - v_curr) / dz
                    dw_dz = (velocity[idx_to_node[(i, j, k+1)], 2] - w_curr) / dz

        advection_accel[node_1d_idx, 0] = u_curr * du_dx + v_curr * du_dy + w_curr * du_dz
        advection_accel[node_1d_idx, 1] = u_curr * dv_dx + v_curr * dv_dy + w_curr * dv_dz
        advection_accel[node_1d_idx, 2] = u_curr * dw_dx + v_curr * dw_dy + w_curr * dw_dz


        # --- Viscous Diffusion Term (mu * Laplacian(u)) ---
        lap_u, lap_v, lap_w = 0.0, 0.0, 0.0

        # d^2/dx^2
        if nx > 1:
            if i > 0 and i < nx - 1:
                lap_u += (velocity[idx_to_node[(i+1,j,k)], 0] - 2*u_curr + velocity[idx_to_node[(i-1,j,k)], 0]) / (dx**2)
                lap_v += (velocity[idx_to_node[(i+1,j,k)], 1] - 2*v_curr + velocity[idx_to_node[(i-1,j,k)], 1]) / (dx**2)
                lap_w += (velocity[idx_to_node[(i+1,j,k)], 2] - 2*w_curr + velocity[idx_to_node[(i-1,j,k)], 2]) / (dx**2)
            elif i == 0 and nx > 1: # Forward difference for second derivative
                lap_u += (velocity[idx_to_node[(i+2,j,k)], 0] - 2*velocity[idx_to_node[(i+1,j,k)], 0] + u_curr) / (dx**2) if nx > 2 else 0.0
                lap_v += (velocity[idx_to_node[(i+2,j,k)], 1] - 2*velocity[idx_to_node[(i+1,j,k)], 1] + v_curr) / (dx**2) if nx > 2 else 0.0
                lap_w += (velocity[idx_to_node[(i+2,j,k)], 2] - 2*velocity[idx_to_node[(i+1,j,k)], 2] + w_curr) / (dx**2) if nx > 2 else 0.0
            elif i == nx - 1 and nx > 1: # Backward difference for second derivative
                lap_u += (u_curr - 2*velocity[idx_to_node[(i-1,j,k)], 0] + velocity[idx_to_node[(i-2,j,k)], 0]) / (dx**2) if nx > 2 else 0.0
                lap_v += (v_curr - 2*velocity[idx_to_node[(i-1,j,k)], 1] + velocity[idx_to_node[(i-2,j,k)], 1]) / (dx**2) if nx > 2 else 0.0
                lap_w += (w_curr - 2*velocity[idx_to_node[(i-1,j,k)], 2] + velocity[idx_to_node[(i-2,j,k)], 2]) / (dx**2) if nx > 2 else 0.0

        # d^2/dy^2
        if ny > 1:
            if j > 0 and j < ny - 1:
                lap_u += (velocity[idx_to_node[(i,j+1,k)], 0] - 2*u_curr + velocity[idx_to_node[(i,j-1,k)], 0]) / (dy**2)
                lap_v += (velocity[idx_to_node[(i,j+1,k)], 1] - 2*v_curr + velocity[idx_to_node[(i,j-1,k)], 1]) / (dy**2)
                lap_w += (velocity[idx_to_node[(i,j+1,k)], 2] - 2*w_curr + velocity[idx_to_node[(i,j-1,k)], 2]) / (dy**2)
            elif j == 0 and ny > 1:
                lap_u += (velocity[idx_to_node[(i,j+2,k)], 0] - 2*velocity[idx_to_node[(i,j+1,k)], 0] + u_curr) / (dy**2) if ny > 2 else 0.0
                lap_v += (velocity[idx_to_node[(i,j+2,k)], 1] - 2*velocity[idx_to_node[(i,j+1,k)], 1] + v_curr) / (dy**2) if ny > 2 else 0.0
                lap_w += (velocity[idx_to_node[(i,j+2,k)], 2] - 2*velocity[idx_to_node[(i,j+1,k)], 2] + w_curr) / (dy**2) if ny > 2 else 0.0
            elif j == ny - 1 and ny > 1:
                lap_u += (u_curr - 2*velocity[idx_to_node[(i,j-1,k)], 0] + velocity[idx_to_node[(i,j-2,k)], 0]) / (dy**2) if ny > 2 else 0.0
                lap_v += (v_curr - 2*velocity[idx_to_node[(i,j-1,k)], 1] + velocity[idx_to_node[(i,j-2,k)], 1]) / (dy**2) if ny > 2 else 0.0
                lap_w += (w_curr - 2*velocity[idx_to_node[(i,j-1,k)], 2] + velocity[idx_to_node[(i,j-2,k)], 2]) / (dy**2) if ny > 2 else 0.0

        # d^2/dz^2
        if nz > 1:
            if k > 0 and k < nz - 1:
                lap_u += (velocity[idx_to_node[(i,j,k+1)], 0] - 2*u_curr + velocity[idx_to_node[(i,j,k-1)], 0]) / (dz**2)
                lap_v += (velocity[idx_to_node[(i,j,k+1)], 1] - 2*v_curr + velocity[idx_to_node[(i,j,k-1)], 1]) / (dz**2)
                lap_w += (velocity[idx_to_node[(i,j,k+1)], 2] - 2*w_curr + velocity[idx_to_node[(i,j,k-1)], 2]) / (dz**2)
            elif k == 0 and nz > 1:
                lap_u += (velocity[idx_to_node[(i,j,k+2)], 0] - 2*velocity[idx_to_node[(i,j,k+1)], 0] + u_curr) / (dz**2) if nz > 2 else 0.0
                lap_v += (velocity[idx_to_node[(i,j,k+2)], 1] - 2*velocity[idx_to_node[(i,j,k+1)], 1] + v_curr) / (dz**2) if nz > 2 else 0.0
                lap_w += (velocity[idx_to_node[(i,j,k+2)], 2] - 2*velocity[idx_to_node[(i,j,k+1)], 2] + w_curr) / (dz**2) if nz > 2 else 0.0
            elif k == nz - 1 and nz > 1:
                lap_u += (u_curr - 2*velocity[idx_to_node[(i,j,k-1)], 0] + velocity[idx_to_node[(i,j,k-2)], 0]) / (dz**2) if nz > 2 else 0.0
                lap_v += (v_curr - 2*velocity[idx_to_node[(i,j-1,k)], 1] + velocity[idx_to_node[(i,j-2,k)], 1]) / (dz**2) if nz > 2 else 0.0
                lap_w += (w_curr - 2*velocity[idx_to_node[(i,j,k-1)], 2] + velocity[idx_to_node[(i,j-2,k)], 2]) / (dz**2) if nz > 2 else 0.0

        diffusion_accel[node_1d_idx, 0] = viscosity * lap_u
        diffusion_accel[node_1d_idx, 1] = viscosity * lap_v
        diffusion_accel[node_1d_idx, 2] = viscosity * lap_w

    # Update tentative velocity (u* = u_n + dt * (adv + diff))
    u_tentative = velocity + dt * (-advection_accel + diffusion_accel) # Note the negative sign for advection in Euler form

    # --- Step 2: Pressure Projection (Enforce Incompressibility) ---
    phi = np.zeros(num_nodes) # Pressure correction field
    divergence_ut = np.zeros(num_nodes)

    # Calculate divergence of tentative velocity (source term for Poisson equation)
    for node_1d_idx in range(num_nodes):
        i, j, k = node_to_idx[node_1d_idx]
        
        # d/dx (u_x)
        div_x = 0.0
        if nx > 1:
            if i > 0 and i < nx - 1:
                div_x = (u_tentative[idx_to_node[(i+1, j, k)], 0] - u_tentative[idx_to_node[(i-1, j, k)], 0]) / (2 * dx)
            elif i == 0: # Forward difference
                div_x = (u_tentative[idx_to_node[(i+1, j, k)], 0] - u_tentative[node_1d_idx, 0]) / dx
            elif i == nx - 1: # Backward difference
                div_x = (u_tentative[node_1d_idx, 0] - u_tentative[idx_to_node[(i-1, j, k)], 0]) / dx

        # d/dy (u_y)
        div_y = 0.0
        if ny > 1:
            if j > 0 and j < ny - 1:
                div_y = (u_tentative[idx_to_node[(i, j+1, k)], 1] - u_tentative[idx_to_node[(i, j-1, k)], 1]) / (2 * dy)
            elif j == 0:
                div_y = (u_tentative[idx_to_node[(i, j+1, k)], 1] - u_tentative[node_1d_idx, 1]) / dy
            elif j == ny - 1:
                div_y = (u_tentative[node_1d_idx, 1] - u_tentative[idx_to_node[(i, j-1, k)], 1]) / dy

        # d/dz (u_z)
        div_z = 0.0
        if nz > 1:
            if k > 0 and k < nz - 1:
                div_z = (u_tentative[idx_to_node[(i, j, k+1)], 2] - u_tentative[idx_to_node[(i, j, k-1)], 2]) / (2 * dz)
            elif k == 0:
                div_z = (u_tentative[idx_to_node[(i, j, k+1)], 2] - u_tentative[node_1d_idx, 2]) / dz
            elif k == nz - 1:
                div_z = (u_tentative[node_1d_idx, 2] - u_tentative[idx_to_node[(i, j, k-1)], 2]) / dz

        divergence_ut[node_1d_idx] = div_x + div_y + div_z

    poisson_rhs = (density / dt) * divergence_ut

    # Successive Over-Relaxation (SOR) iteration for Poisson equation (del^2(phi) = S)
    num_poisson_iterations = 100 # Increased iterations for better convergence
    omega = 1.7 # Over-relaxation factor (typically 1.0 to 2.0)

    phi_new = np.copy(phi)

    for _iter in range(num_poisson_iterations):
        for node_1d_idx in range(num_nodes):
            i, j, k = node_to_idx[node_1d_idx]
            
            phi_sum_neighbors = 0.0
            coeff_diag = 0.0

            # X-direction
            if i > 0: phi_sum_neighbors += phi_new[idx_to_node[(i-1, j, k)]] / (dx**2) # Use new values if available (Gauss-Seidel like)
            if i < nx - 1: phi_sum_neighbors += phi[idx_to_node[(i+1, j, k)]] / (dx**2) # Use old values
            if nx > 1: coeff_diag += 2 / (dx**2) if i > 0 and i < nx - 1 else 1 / (dx**2)

            # Y-direction
            if j > 0: phi_sum_neighbors += phi_new[idx_to_node[(i, j-1, k)]] / (dy**2)
            if j < ny - 1: phi_sum_neighbors += phi[idx_to_node[(i, j+1, k)]] / (dy**2)
            if ny > 1: coeff_diag += 2 / (dy**2) if j > 0 and j < ny - 1 else 1 / (dy**2)

            # Z-direction (if 3D)
            if nz > 1:
                if k > 0: phi_sum_neighbors += phi_new[idx_to_node[(i, j, k-1)]] / (dz**2)
                if k < nz - 1: phi_sum_neighbors += phi[idx_to_node[(i, j, k+1)]] / (dz**2)
                coeff_diag += 2 / (dz**2) if k > 0 and k < nz - 1 else 1 / (dz**2)

            if coeff_diag > 1e-12:
                # SOR update rule
                phi_new[node_1d_idx] = (1 - omega) * phi[node_1d_idx] + \
                                       omega * ((phi_sum_neighbors - poisson_rhs[node_1d_idx]) / coeff_diag)
            else:
                phi_new[node_1d_idx] = 0.0
        
        phi = np.copy(phi_new) # Update for next iteration

    # --- Step 3: Correct Velocity and Update Pressure ---
    for node_1d_idx in range(num_nodes):
        i, j, k = node_to_idx[node_1d_idx]

        dphi_dx, dphi_dy, dphi_dz = 0.0, 0.0, 0.0

        # d/dx
        if nx > 1:
            if i > 0 and i < nx - 1:
                dphi_dx = (phi[idx_to_node[(i+1, j, k)]] - phi[idx_to_node[(i-1, j, k)]]) / (2 * dx)
            elif i == 0:
                dphi_dx = (phi[idx_to_node[(i+1, j, k)]] - phi[node_1d_idx]) / dx
            elif i == nx - 1:
                dphi_dx = (phi[node_1d_idx] - phi[idx_to_node[(i-1, j, k)]]) / dx

        # d/dy
        if ny > 1:
            if j > 0 and j < ny - 1:
                dphi_dy = (phi[idx_to_node[(i, j+1, k)]] - phi[idx_to_node[(i, j-1, k)]]) / (2 * dy)
            elif j == 0:
                dphi_dy = (phi[idx_to_node[(i, j+1, k)]] - phi[node_1d_idx]) / dy
            elif j == ny - 1:
                dphi_dy = (phi[node_1d_idx] - phi[idx_to_node[(i, j-1, k)]]) / dy

        # d/dz
        if nz > 1:
            if k > 0 and k < nz - 1:
                dphi_dz = (phi[idx_to_node[(i, j, k+1)]] - phi[idx_to_node[(i, j, k-1)]]) / (2 * dz) # Corrected
            elif k == 0:
                dphi_dz = (phi[idx_to_node[(i, j, k+1)]] - phi[node_1d_idx]) / dz
            elif k == nz - 1:
                dphi_dz = (phi[node_1d_idx] - phi[idx_to_node[(i, j, k-1)]]) / dz # Corrected

        # Apply correction to velocity (u_new = u_tentative - dt * (1/rho) * grad(phi))
        u_tentative[node_1d_idx, 0] -= dt * dphi_dx / density
        u_tentative[node_1d_idx, 1] -= dt * dphi_dy / density
        u_tentative[node_1d_idx, 2] -= dt * dphi_dz / density
    
    # Update pressure (new_pressure = old_pressure + phi * density / dt)
    new_pressure = pressure + phi * (density / dt) 

    return u_tentative, new_pressure


def run_simulation(json_filename):
    """
    Runs the fluid simulation based on simplified Navier-Stokes equations.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data", "testing-input-output")
    
    input_filepath = os.path.join(data_dir, json_filename)
    data = load_json(input_filepath)

    mesh_data = data["mesh"]
    fluid_properties = data["fluid_properties"]
    boundary_conditions = data["boundary_conditions"]
    simulation_params = data["simulation_parameters"]

    time_step = simulation_params["time_step"]
    total_time = simulation_params["total_time"]
    num_steps = int(total_time / time_step)

    # --- Dynamic Grid Dimension Derivation ---
    num_nodes_from_json = mesh_data["nodes"]
    
    # Estimate domain size from boundary_faces if available, otherwise use default
    domain_min = [float('inf'), float('inf'), float('inf')]
    domain_max = [float('-inf'), float('-inf'), float('-inf')]
    
    # If boundary_faces has node coordinates, use them to estimate domain_size
    if "boundary_faces" in mesh_data and mesh_data["boundary_faces"]:
        for face_info in mesh_data["boundary_faces"]:
            for node_coords_str in face_info["nodes"].values():
                coords = np.array(node_coords_str)
                domain_min = np.minimum(domain_min, coords)
                domain_max = np.maximum(domain_max, coords)
        
        # Calculate domain_size based on min/max coords
        domain_size = tuple(domain_max - domain_min)
        # Ensure domain_size components are not zero if there's only one point in that dimension
        for i in range(3):
            if domain_size[i] < 1e-9: # If essentially zero, assume a unit size for that dimension if it's a 1-node dimension
                domain_size[i] = 1.0 # Or some sensible default, assuming a 1x1x1 unit cube if no extent
    else:
        domain_size = (1.0, 1.0, 1.0) # Default if no coordinate data

    desired_grid_dims = find_optimal_grid_dimensions(num_nodes_from_json)
    print(f"Automatically determined grid dimensions: {desired_grid_dims}")
    
    num_nodes_actual, nodes_coords, grid_shape, dx, dy, dz, node_to_idx, idx_to_node = \
        create_structured_grid_info(grid_dims=desired_grid_dims, domain_size=domain_size)

    # Overwrite the num_nodes from JSON with the actual count from the grid
    mesh_data["nodes"] = num_nodes_actual 

    mesh_info = {
        "nodes": mesh_data["nodes"],
        "nodes_coords": nodes_coords,
        "grid_shape": grid_shape,
        "dx": dx, "dy": dy, "dz": dz,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node
    }

    # Initialize fields using inlet conditions
    initial_velocity = np.array(boundary_conditions["inlet"]["velocity"])
    initial_pressure = boundary_conditions["inlet"]["pressure"]
    velocity, pressure = initialize_fields(mesh_info["nodes"], initial_velocity, initial_pressure)

    # --- CFL Condition Enforcement ---
    max_inlet_velocity_magnitude = np.linalg.norm(initial_velocity)
    # Estimate a characteristic velocity for CFL condition. Using inlet velocity is a simple start.
    # In a more advanced solver, this would be the maximum velocity in the domain.
    if max_inlet_velocity_magnitude > 1e-9: # Avoid division by zero
        min_dx_dy_dz = min(dx, dy, dz)
        # Handle cases where dx, dy, dz might be 0 due to 1D/2D configuration.
        # If a dimension is 1, its corresponding dx/dy/dz is essentially infinite for derivatives,
        # so it doesn't limit the CFL condition. Filter out effectively infinite values.
        finite_spacings = [s for s in [dx, dy, dz] if s > 1e-9]
        if not finite_spacings: # All dimensions are effectively 1 (e.g., a single point)
            cfl_dt_limit = float('inf')
        else:
            cfl_dt_limit = min(finite_spacings) / max_inlet_velocity_magnitude
        
        # A typical CFL number for explicit Euler is <= 1.0. For stability, let's target 0.5.
        target_cfl_num = 0.5 
        recommended_dt = target_cfl_num * cfl_dt_limit

        if time_step > recommended_dt:
            print(f"⚠️ Warning: Time step ({time_step:.4f}s) exceeds recommended CFL limit ({recommended_dt:.4f}s).")
            print(f"    This may lead to numerical instability. Consider reducing time_step.")
            # Optionally, you could force time_step = recommended_dt here.
            # For this simplified solver, we'll just warn.
    else:
        print("Note: Cannot calculate CFL limit (inlet velocity is zero). Time step stability not checked.")


    all_velocities = []
    all_pressures = []
    time_points = []

    print(f"Simulation starting for {num_steps} steps (Total Time: {total_time}s, Time Step: {time_step}s)")
    print(f"Derived 3D grid shape: {grid_shape} with dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    print(f"Total nodes: {mesh_info['nodes']}")

    for step in range(num_steps):
        current_time = (step + 1) * time_step
        
        apply_boundary_conditions(velocity, pressure, boundary_conditions, mesh_info)
        
        velocity, pressure = compute_next_step(velocity, pressure, mesh_info, fluid_properties, time_step)
        
        all_velocities.append(velocity.tolist())
        all_pressures.append(pressure.tolist())
        time_points.append(current_time)

        if (step + 1) % 10 == 0 or step == num_steps - 1: # Print progress
            print(f"Step {step+1}/{num_steps} (Time: {current_time:.2f}s): "
                  f"Avg Velocity = {np.mean(velocity, axis=0)}, Avg Pressure = {np.mean(pressure):.2f}")

    output_filename = os.path.join(data_dir, "navier_stokes_results.json")
    with open(output_filename, "w") as output_file:
        json.dump({
            "time_points": time_points,
            "velocity_history": all_velocities,
            "pressure_history": all_pressures,
            "mesh_info": { # Include mesh info for post-processing
                "nodes": mesh_info["nodes"],
                "nodes_coords": mesh_info["nodes_coords"].tolist(),
                "grid_shape": mesh_info["grid_shape"],
                "dx": mesh_info["dx"], "dy": mesh_info["dy"], "dz": mesh_info["dz"]
            }
        }, output_file, indent=4)

    print(f"✅ Simulation completed. Results saved to {output_filename}")

# Run the solver
# Ensure the input JSON file exists in data/testing-input-output/ relative to src/
run_simulation("fluid_simulation_input.json")
