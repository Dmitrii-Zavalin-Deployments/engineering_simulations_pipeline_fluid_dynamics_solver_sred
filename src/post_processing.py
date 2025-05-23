import json

def export_results(mesh, U, P, k, epsilon, sim_settings, filename):
    results = {
        "simulation_info": {
            "solver": "Navier-Stokes",
            "turbulence_model": sim_settings["turbulence_model"],
            "time_step": sim_settings["suggested_time_step"]
        },
        "units": {
            "coordinates": "m",
            "velocity": "m/s",
            "pressure": "Pa"
        },
        "coordinates": [cell.coordinates for cell in mesh.cells],
        "velocity": [cell.velocity.tolist() for cell in mesh.cells],
        "pressure": [cell.pressure for cell in mesh.cells]
    }

    with open(filename, "w") as file:
        json.dump(results, file, indent=4)



