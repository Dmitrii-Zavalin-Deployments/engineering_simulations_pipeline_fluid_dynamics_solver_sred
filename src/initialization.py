def initialize_simulation(mesh, boundary_data):
    sim_settings = boundary_data["simulation_settings"]
    U_ref, turbulence_intensity, length_scale = 1.0, 0.05, 0.1

    for cell in mesh.cells:
        cell.velocity = np.zeros(3)
        cell.pressure = boundary_data["inlet_boundary"]["pressure"]
        cell.k = 1.5 * (U_ref * turbulence_intensity) ** 2
        cell.epsilon = (cell.k ** 1.5) / length_scale

    return sim_settings



