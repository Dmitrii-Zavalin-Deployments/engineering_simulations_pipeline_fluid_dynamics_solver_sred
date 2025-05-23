from scipy.sparse.linalg import bicgstab

def solve_turbulence_equations(k, epsilon, U, mesh):
    A_k, b_k, A_epsilon, b_epsilon = assemble_turbulence_matrix(k, epsilon, U, mesh)

    k_new, _ = bicgstab(A_k, b_k)
    epsilon_new, _ = bicgstab(A_epsilon, b_epsilon)

    return k_new, epsilon_new



