def adjust_time_step(sim_settings, U, mesh):
    CFL = sim_settings["CFL_condition"]
    dt_max = sim_settings["suggested_time_step"] * 5
    dt_min = sim_settings["suggested_time_step"] / 10

    max_velocity = max(np.linalg.norm(cell.velocity) for cell in mesh.cells)
    min_cell_size = min(cell.size for cell in mesh.cells)

    dt_new = CFL * (min_cell_size / max_velocity)
    return max(min(dt_new, dt_max), dt_min)



