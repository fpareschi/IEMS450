import numpy as np

class ObjectiveFunction(object):

    def __init__(self, n):
        self.n = n

    def f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape((self.n, ))

        return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

    def delta_f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)

        x = x.reshape((self.n, ))

        delta1 = 400*(x[0]**3) - 400*x[0]*x[1] + 2*x[0] - 2
        delta2 = 200*(x[1]-x[0]**2)

        return np.array([delta1, delta2])

    def d_delta_f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape((self.n, ))

        d11 = 1200*(x[0]**2) - 400*x[1] + 2
        d12 = - 400*x[0]
        d21 = - 400*x[0]
        d22 = 200
        return np.array([[d11, d12], [d21, d22]])


class LinearSearch(object):

    def __init__(self,  rho, c, alpha_bar, n):
        """
        rho: backtracking reduction parameter
        c: sufficient decrease parameter
        alpha_bar: initial steplenght in backtracking

        """
        self.rho = rho
        self.c = c
        self.alpha_bar = alpha_bar
        self.objective = ObjectiveFunction(n)


    def compute_direction(self, x_k, H_k, method='Steepest'):
        implemented_method = ['Steepest', 'Newton', 'BFGS']
        assert method in implemented_method

        if method=='Steepest':
            pk = - self.objective.delta_f(x_k)

        elif method=='Newton':
            pk = - np.linalg.inv(self.objective.d_delta_f(x_k)) @ self.objective.delta_f(x_k)

        elif method == 'BFGS':
            pk = - H_k @ self.objective.delta_f(x_k)

        return pk

    def update_H(self, x_k, H_k, alpha_k):

        x_k1 = x_k + self.compute_direction(x_k, H_k, method='BFGS') * alpha_k

        s_k = (x_k1 - x_k)
        y_k = (self.objective.delta_f(x_k1) - self.objective.delta_f(x_k))
        rho_k = 1 / (y_k.T @ s_k)

        s_k = s_k.reshape((self.objective.n, 1))
        y_k = y_k.reshape((self.objective.n, 1))

        H_k1 = (np.identity(x_k.shape[0]) - rho_k * (s_k @ y_k.T)) @ H_k @ (np.identity(x_k.shape[0]) -
                                                                        rho_k * (y_k @ s_k.T)) + rho_k * (s_k @ s_k.T)

        return H_k1

    def backtracking(self, x_k, p_k):

        upper_f = self.objective.f(x_k) + self.c*self.alpha_bar*(self.objective.delta_f(x_k).T @ p_k)
        x_k1 = x_k + self.alpha_bar * p_k
        fk1 = self.objective.f(x_k1)
        j = 1
        alpha_j = self.alpha_bar

        while fk1 > upper_f:
            x_k1 = x_k + alpha_j * p_k
            tol_fval, fk1 = self.objective.f(x_k) + self.c * alpha_j * (self.objective.delta_f(x_k).T @ p_k), self.objective.f(x_k1)
            print(tol_fval)
            alpha_j = self.rho * alpha_j
            j += 1

        return alpha_j, j


tol = 1e-6
x_0 = np.array([-1.2, 1.0])
x_k = x_0

alpha_bar = 1
rho = 0.5
c = 10e-4
obj = ObjectiveFunction(2)
H_0 = np.linalg.inv(obj.d_delta_f(x_0))
H_k = H_0

i = 0
crit = 1000
print('iter \\  f \\ ||p_k|| \\ alpha \\ #func \\ ||grad_f||')

while crit > tol:

    LS_inst = LinearSearch(rho, c, alpha_bar, 2)
    p_k = LS_inst.compute_direction(x_k, H_k, method='BFGS')
    alpha_k, fval = LS_inst.backtracking(x_k, p_k)

    x_k1 = x_k + p_k * alpha_k
    H_k = LS_inst.update_H(x_k, H_k, alpha_k)

    crit = np.abs(obj.delta_f(x_k1)).max()
    descent = np.abs(p_k).max()
    x_k = x_k1

    i += 1
    print(i , round(obj.f(x_k1), 20),  round(descent, 10), round(alpha_k, 4), round(fval, 4), round(crit, 8))

    if i > 50:
        break
