def correct_velocity(U, P_prime, mesh):
    for cell in mesh.cells:
        pressure_term = gradient(P_prime, cell)
        U[cell] -= (dt / ap) * pressure_term



