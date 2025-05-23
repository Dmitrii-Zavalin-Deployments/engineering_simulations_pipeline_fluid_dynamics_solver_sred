def check_convergence(U, P, k, epsilon, tolerance):
    residual_u = np.linalg.norm(U)
    residual_p = np.linalg.norm(P)
    residual_k = np.linalg.norm(k)
    residual_epsilon = np.linalg.norm(epsilon)

    return all(residual < tolerance for residual in [residual_u, residual_p, residual_k, residual_epsilon])



