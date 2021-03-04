import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import ldl
from scipy.linalg import lu

def _print_first_iter():
    #np.set_printoptions(precision=2,suppress=True)
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


    print(' iter '
            + '         f '
            + '           x   '
            + '    Active C   '
            + '      lambdas      '
            + '      p_(k)    '
            + '   alpha_(k)'
            )

    pass

def _print_each_iter(i, x_k_, p_k_, alpha_, lambdas_, active_):

    print(f'{i:4d}  '
           + f'   {obj_instance.f(x_k[0], x_k[1]):.2E} '
           + f'  {x_k_}'
           + f'   {active_}   '
           + f'   {lambdas_}  '
           + f'   {p_k_}'
           + f'   {alpha_:.2E}'
           )
    pass

class ObjectiveFunction(object):

    def __init__(self):
        pass

    def f(self,x_1, x_2):
        return - 6*x_1 - 4*x_2 + 13 + x_1**2 + x_2**2

    def delta_f(self, x_1, x_2):
        return np.array([-6 + 2*x_1, -4 + 2*x_2])


class ActiveSet(object):

    def __init__(self, G, Afull, b):

        self.G = G
        self.Afull = Afull
        self.b  = b

        self.obj = ObjectiveFunction()

    def compute_p(self, active_constraint, x_k):
        g_k = self.obj.delta_f(x_k[0], x_k[1])
        if not active_constraint:
            assert len(active_constraint) == 0
            p = np.linalg.solve(self.G, g_k)

        else:
            assert  len(active_constraint) > 0

            A = self.Afull[active_constraint, :]

            RHS = np.array([*g_k, *np.zeros(len(active_constraint))])
            K_u = np.hstack([self.G, A.T])
            K_l = np.hstack([A, np.zeros((A.shape[0], A.shape[0]))])
            K = np.vstack([K_u, K_l])

            # Perfom LU factorization and solve system
            P, L, U = lu(K, permute_l=False)

            btilde = P @ RHS
            z = np.linalg.solve(L, btilde)
            p = np.linalg.solve(U, z)

        return -p[:2]


    def compute_lambda(self, active_constraint, x_k):

        if len(active_constraint)>1:
            A = self.Afull[active_constraint, :]
            Lambda = np.linalg.solve(A, self.obj.delta_f(x_k[0], x_k[1]))

        else:
            A = self.Afull[active_constraint, :].flatten()

            if A[0] !=0:
                Lambda = self.obj.delta_f(x_k[0], x_k[1])[0]/A[0]
            else:
                Lambda = self.obj.delta_f(x_k[0], x_k[1])[1] / A[1]

        return np.array([Lambda]).flatten()

    def compute_alpha(self, active_constraints, x_k, p_k):
        not_active = np.setdiff1d([0, 1, 2], active_constraints)

        potential_block = []
        for i in not_active:
            if self.Afull[i, :].T @ p_k<0:
                potential_block.append((i, (self.b[i] - self.Afull[i, :] @ x_k)/(self.Afull[i,:] @ p_k)))

        if not potential_block:
            alpha = 1
            blocking = []
        else:
            blocking_id, alpha_pot = min(potential_block, key=lambda t: t[1])
            alpha = min(alpha_pot, 1)
            blocking_id = int(blocking_id)
            if alpha == alpha_pot:
                blocking = not_active[blocking_id]
            else:
                blocking = []

        return alpha, blocking


# Solve Problem Graphically
obj_instance = ObjectiveFunction()

delta = 0.01
x1 = np.arange(-3, 4, delta)
x2 = np.arange(-3, 4, delta)

X1, X2 = np.meshgrid(x1, x2)
Z = -obj_instance.f(X1, X2)

fig, ax = plt.subplots()
ax.scatter(2, 1, color='black')
CS = ax.contour(X1, X2, Z, levels=70)
ax.clabel(CS, inline=1, fontsize=5)
ax.set_title('Graphical Solution')
ax.fill_between(x1[(x1>0) & (x1<3)], 0, 3-x1[(x1>0) & (x1<3)], facecolor='g', alpha=0.4)
ax.set_xlim([-1, 4])
ax.set_ylim([-1, 4])
plt.show()



G = 2*np.identity(2)
Afull = np.array([[-1, -1], [1, 0], [0,1]])
b = np.array([-3, 0, 0])

ac_inst = ActiveSet(G, Afull, b)


x_0 = np.array([0.0, 0.0])
active = [1,2]

x_k = x_0
it = 0
lamb = np.zeros(1)
_print_first_iter()
while it < 5:
    _print_each_iter(it, x_k, p_k, alpha, lamb, active)
    p_k = ac_inst.compute_p(active, x_k)

    if np.linalg.norm(p_k) < 1e-5:
        lamb = ac_inst.compute_lambda(active, x_k)
        if np.min(lamb) >= 0:
            x_star = x_k
            break
        else:
            if lamb.shape[0] == 0:
                active = []
            else:
                lamb_hat = np.min(lamb)
                i = list(np.argwhere((lamb == lamb_hat))[0])
                cons_rem = active[i[0]]
                active = list(np.setdiff1d(active, cons_rem))
    else:
        alpha, block = ac_inst.compute_alpha(active, x_k, p_k)
        x_k = x_k + alpha*p_k

        if type(block) == np.int32:
            active.append(block)
            active = sorted(active)


    it +=1




