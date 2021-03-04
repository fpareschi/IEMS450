import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import ldl
from scipy.linalg import lu


def f(x_1, x_2):
    return - 6*x_1 - 4*x_2 + 13 + x_1**2 + x_2**2


def delta_f(x_1, x_2):
    return np.array([-6 + 2*x_1, -4 + 2*x_2])

delta = 0.01
x1 = np.arange(-3, 4, delta)
x2 = np.arange(-3, 4, delta)

X1, X2 = np.meshgrid(x1, x2)

Z = -f(X1, X2)

fig, ax = plt.subplots()
ax.scatter(2, 1, color='black')
CS = ax.contour(X1, X2, Z, levels=70)
ax.clabel(CS, inline=1, fontsize=5)
ax.set_title('Graphical Solution')
ax.fill_between(x1[(x1>0) & (x1<3)], 0, 3-x1[(x1>0) & (x1<3)], facecolor='g', alpha=0.4)
ax.set_xlim([-1, 4])
ax.set_ylim([-1, 4])
plt.show()


x_0 = np.array([2, 0])

G = 2*np.identity(2)
Afull = np.array([[-1, -1], [1, 0], [0,1]])
b = np.array([-3, 0, 0])
#RHSfull = np.array([6,4,-3, 0, 0])

active =[1,2]


def compute_p(active_constraint, x_k):
    A = Afull[active_constraint, :]

    g_k = delta_f(x_k[0], x_k[1])
    RHS = np.array([*g_k, *np.zeros(len(active_constraint))])

    K_u = np.hstack([G, A.T])
    K_l = np.hstack([A, np.zeros((A.shape[0], A.shape[0]))])
    K = np.vstack([K_u, K_l])

    # Perfom LU factorization and solve system
    P, L, U = lu(K, permute_l=False)

    btilde = P @ RHS
    z = np.linalg.solve(L, btilde)
    p = np.linalg.solve(U, z)

    return -p[:2]


def compute_lambda(active_constraint, x_k):

    if len(active_constraint)>1:
        A = Afull[active_constraint, :]
        Lambda = np.linalg.solve(A, delta_f(x_k[0], x_k[1]))
        print('la')
    else:
        A = Afull[active_constraint, :].flatten()
        if A[0] !=0:
            Lambda = delta_f(x_k[0], x_k[1])[0]/A[0]
        else:
            Lambda = delta_f(x_k[0], x_k[1])[1] / A[1]

    return Lambda

def compute_alpha(active_constraints, x_k, p_k):
    not_active = np.setdiff1d([0,1,2],active_constraints)
    print(not_active)

    alpha_pot = []
    for i in not_active:
        if Afull[i, :].T @ p_k<0:
            alpha_pot.append((b[i] - Afull[i, :].T @ x_k)/(Afull[i,:].T @ p_k))

    alpha_pot = min(alpha_pot)
    alpha = min(alpha_pot, 1)
    return alpha
