import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import ldl
from scipy.linalg import lu

def f(x_1, x_2):
    return - 6*x_1 - 4*x_2 + 13 + x_1**2 + x_2**2

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
RHSfull = np.array([6,4,-3, 0, 0])

active =[1,2]


def compute_p(active_constraint):
    A = Afull[active_constraint, :]

    K_u = np.hstack([G, A.T])
    K_l = np.hstack([A, np.zeros((A.shape[0], A.shape[0]))])

    K = np.vstack([K_u, K_l])
    RHS = np.array([*RHSfull[:2], *RHSfull[2:][active_constraint]])

    # Perfom LU factorization and solve system
    P, L, U = lu(K, permute_l=False)

    btilde = P @ RHS
    z = np.linalg.solve(L, btilde)
    x = np.linalg.solve(U, z)

    return x

