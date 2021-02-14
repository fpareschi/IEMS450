import numpy as np

def set_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = 1/(i + j + 1)
    return A



def update_step_cg(x_k, p_k, r_k,  A):

    alpha_k = (r_k**2).sum()/(p_k.T @ A @ p_k)
    x_k1 = x_k + p_k * alpha_k
    r_k1 = r_k + alpha_k * (A @ p_k)
    beta_k1 = (r_k1 ** 2).sum()/(r_k**2).sum()
    p_k1 = -r_k1 + beta_k1 * p_k

    return x_k1, r_k1, p_k1


def conjugate_gradient(N):
    b = np.ones(N)
    A = set_matrix(N)
    tol = 1e-5

    x_0 = np.zeros(N)
    r_0 = A @ x_0 - b
    p_0 = -r_0

    x_old = x_0
    r_old = r_0
    p_old = p_0
    crit = 100
    i = 1
    while crit>tol:
        x_new, r_new, p_new = update_step_cg(x_old, p_old, r_old, A)
        crit = np.linalg.norm(x_new - x_old)

        x_old = x_new.copy()
        r_old = r_new.copy()
        p_old = p_new.copy()
        i += 1

    print('Conjugate Gradient with %i dimensions, converged after %i iterations' % (N, i))
    pass

for n in [5, 8, 12, 20]:
    conjugate_gradient(n)