def initialize_simulation(mesh, boundary_data):
    """ Initializes velocity, pressure, and turbulence variables for each face. """
    sim_settings = boundary_data["simulation_settings"]

    # Initialize velocity, pressure, and turbulence properties for each face
    for face_id, face in mesh.faces.items():
        face["velocity"] = [0.0, 0.0, 0.0]  # Initialize velocity field
        face["pressure"] = boundary_data["inlet_boundary"]["pressure"] if face["type"] == "inlet" else None
        face["k"] = 0.0  # Initialize turbulent kinetic energy
        face["epsilon"] = 0.0  # Initialize turbulence dissipation rate

    return sim_settings



