import numpy as np

class ObjectiveFunction(object):

    def __init__(self, n):
        self.n = n

    def f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape((self.n, ))

        if self.n == 2:
            return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

        else:
            f = (100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2).sum()
            return f

    def delta_f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)

        x = x.reshape((self.n, ))

        if self.n == 2:
            delta1 = 400*(x[0]**3) - 400*x[0]*x[1] + 2*x[0] - 2
            delta2 = 200*(x[1]-x[0]**2)

            return np.array([delta1, delta2])

        else:
            deltaf = np.zeros(self.n)
            for i in range(self.n):
                if i == 0:
                    deltaf[i] = 400*(x[i]**3) - 400*x[i]*x[i+1] + 2*x[i] - 2
                elif (i>0) and (i<self.n-1):
                    deltaf[i] = 400 * (x[i] ** 3) - 400 * x[i] * x[i + 1] + 2 * x[i] - 2 + 200*(x[i] - x[i-1]**2)
                else:
                    assert i == self.n - 1
                    deltaf[i] = 200*(x[i] - x[i-1]**2)

            return deltaf

    def d_delta_f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape((self.n, ))

        if self.n == 2:
            d11 = 1200*(x[0]**2) - 400*x[1] + 2
            d12 = - 400*x[0]
            d21 = - 400*x[0]
            d22 = 200
            return np.array([[d11, d12], [d21, d22]])

        else:
            ddeltaf = np.zeros((self.n, self.n))
            for i in range(self.n):
                if i == 0:
                    ddeltaf[i, i] = -400*x[i+1] + 1200*x[i]**2 + 2
                    ddeltaf[i, i+1] = -400*x[i]
                elif (i>0) and (i<self.n-1):
                    ddeltaf[i, i-1] = -400*x[i-1]
                    ddeltaf[i, i] = 1200*x[i]**2 - 400*x[i+1] + 202
                    ddeltaf[i, i+1] = -400*x[i]
                else:
                    assert i == self.n - 1
                    ddeltaf[i, i-1] = -400*x[i-1]
                    ddeltaf[i, i] = 200
            return ddeltaf

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
            upper_f, fk1 = self.objective.f(x_k) + self.c * alpha_j * (self.objective.delta_f(x_k).T @ p_k), self.objective.f(x_k1)
            alpha_j = self.rho * alpha_j
            j += 1

        return alpha_j, j

    def main_loop(self, x0, tol, max_it, method='Steepest'):
        implemented_method = ['Steepest', 'Newton', 'BFGS']
        assert method in implemented_method

        H0 = np.linalg.inv(self.objective.d_delta_f(x0))
        Hk = H0
        xk = x0

        i = 0
        crit = 1000

        #print('iter \\  f \\ ||p_k|| \\ alpha \\ #func \\ ||grad_f||')
        _print_first_iter()

        while crit > tol:

            pk = self.compute_direction(xk, Hk, method=method)
            alpha_k, fval = self.backtracking(xk, pk)

            xk1 = xk + pk * alpha_k
            Hk = self.update_H(xk, Hk, alpha_k)

            crit = np.abs(self.objective.delta_f(xk1)).max()
            grad_norm = np.abs(pk).max()
            xk = xk1

            i += 1

            if i <= 50:
                _print_each_iter(i, self.objective.f(xk1), grad_norm, alpha_k, fval, crit )
            elif i>50:
                if i/100 == np.floor(i/100):
                    _print_each_iter(i, self.objective.f(xk1), grad_norm, alpha_k, fval, crit)
                #print(i, round(self.objective.f(xk1), 20), round(grad_norm, 10), round(alpha_k, 4), round(fval, 4), round(crit, 8))

            if i > max_it:
                break
        _print_each_iter(i, self.objective.f(xk1), grad_norm, alpha_k, fval, crit)

def _print_first_iter():
    print('iter '
         + 'f            '
         + '||p_k||      '
         + 'alpha   '
         + '#func   '
         + '||grad_f||')

    pass

def _print_each_iter(i, obj,  grad_norm, alpha_k, feval, crit):

    print('%i    %4.04e   %4.04e   %4.04f    %i    %4.04e' % (i, obj,  grad_norm, alpha_k, feval, crit))

    pass

tol = 1e-6
x_0 = np.array([-1.2, 1.0])
max_it = 15000

alpha_bar = 1
rho = 0.5
c = 10e-4

LSinst = LinearSearch(rho, c, alpha_bar, 2)
LSinst.main_loop(x_0, tol, max_it, method='Steepest')


N = 100
x_0n =-np.ones(100)

LSinst = LinearSearch(rho, c, alpha_bar, N)
LSinst.main_loop(x_0n, tol, max_it, method='Steepest')


