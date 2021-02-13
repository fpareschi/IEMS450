import numpy as np
import scipy.optimize as optimize

class ObjectiveFunction(object):

    def __init__(self, n):
        self.n = n

    def f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape((self.n, ))

        f = 0.5*(x[0] - 1)**3 + 0.5*((x[:-1] - 2*x[1:])**4).sum()
        return f

    def delta_f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)

        x = x.reshape((self.n, ))

        deltaf = np.zeros(self.n)
        for i in range(self.n):
            if i == 0:
                deltaf[i] = x[0] - 1 + 2*(x[0] - 2*x[1])**3

            elif (i>0) and (i<self.n-1):
                deltaf[i] = 2*(x[i] - 2*x[i+1])**3 - 4*(x[i-1] - 2*x[i])**3

            else:
                assert i == self.n - 1
                deltaf[i] = - 4*(x[i-1] - 2*x[i])**3

        return deltaf

    def d_delta_f(self, x):
        assert x.shape == (self.n ,) or x.shape == (self.n,1)
        x = x.reshape((self.n, ))

        ddeltaf = np.zeros((self.n, self.n))
        for i in range(self.n):
            if i == 0:
                ddeltaf[i, i] = 1 + 6*(x[i] - 2*x[i+1])**2
                ddeltaf[i, i+1] = -12*(x[i] - 2*x[i+1])**2
            elif (i>0) and (i<self.n-1):
                ddeltaf[i, i-1] = -12*(x[i-1] - 2*x[i])**2
                ddeltaf[i, i] = 6*(x[i] - 2*x[i+1])**2 + 24*(x[i-1] - 2*x[i])**2
                ddeltaf[i, i+1] = -12*(x[i] - 2*x[i+1])**2
            else:
                assert i == self.n - 1
                ddeltaf[i, i-1] = -12*(x[i-1]- 2*x[i])**2
                ddeltaf[i, i] = 24*(x[i-1] - 2*x[i])**2
        return ddeltaf



class TrustRegion(object):

    def __init__(self, N, delta_hat, eta):
        self.obj = ObjectiveFunction(N)
        self.N = N
        self.delta_hat = delta_hat
        self.eta = eta

    def quadratic_model(self, x_k, p):
        m = self.obj.f(x_k) + (self.obj.delta_f(x_k) * p).sum() + 0.5 * p.T @ self.obj.d_delta_f(x_k) @ p
        return m

    def update_trust_region(self, x_k, p_k, delta_k):
        """
        :param delta_k: current ratio of trust-region.
        :param x_k: current iterate argument
        :param p_k: the direction is computed from solving the subproblem using conjugate gradients.
        :return:
        """
        num = self.obj.f(x_k) - self.obj.f(x_k + p_k)
        den = self.quadratic_model(x_k, np.zeros(self.N)) - self.quadratic_model(x_k, p_k)
        rho_k = num/den

        if rho_k < 0.25:
            delta_k1 = 0.25 * delta_k
        elif rho_k > 0.75 and np.linalg.norm(p_k)==delta_k:
            delta_k1 = np.minimum(2*delta_k, self.delta_hat)
        else:
            delta_k1 = delta_k

        # It rho_k is to small, we don't trust the direction and repeat. Otherwise, use step, although we might adjust.
        if rho_k>self.eta:
            x_k1 = x_k + p_k
        else:
            x_k1 = x_k

        return x_k1, delta_k1


    def solve_tau_norm(self, z_j, d_j, delta):
        x_star =  optimize.fsolve(lambda x: np.linalg.norm(z_j + x*d_j) - delta, np.array([1]))
        return x_star


    def minimize_tau(self, z_j, d_j, x_k, delta):
        if d_j * self.obj.delta_f(x_k) <= 0:
            tau = self.solve_tau_norm(z_j, d_j, delta)
        else:
            tau_candidate = (d_j * (self.obj.delta_f(x_k) - self.obj.d_delta_f(x_k) @ z_j)).sum()/\
                            (d_j.T @ self.obj.d_delta_f(x_k) @ d_j)
            if tau_candidate > 0:
                tau = tau_candidate
            else:
                tau = self.solve_tau_norm(z_j, d_j, delta)

        assert np.linalg.norm(z_j + tau * d_j) <= delta

        return tau

    def step_inner_loop_CG(self, z_j, d_j, r_j, x_k, delta, tol):

        if d_j.T @ self.obj.d_delta_f(x_k) @ d_j <=0:
            #print('Negative curvature')
            tau = self.minimize_tau(z_j, d_j, x_k, delta)
            assert  tau >0

            p_k = z_j + d_j * tau
            return p_k, 1
        else:
            alpha_j = ((r_j**2).sum()/(d_j[:, np.newaxis].T @ self.obj.d_delta_f(x_k) @ d_j[:, np.newaxis])).flatten()
            z_j1 = z_j + alpha_j * d_j
            if np.linalg.norm(z_j1) >= delta:
                #print('Outside tr')
                tau = self.solve_tau_norm(z_j, d_j, delta)
                assert tau > 0
                p_k = z_j + tau * d_j
                assert np.linalg.norm(p_k) <= delta + 1e-5
                return p_k, 1
            else:
                r_j1 = r_j + alpha_j * self.obj.d_delta_f(x_k) @ d_j
                #print(tol, np.linalg.norm(r_j1))
                if np.linalg.norm(r_j1) < tol:
                    #print('Tolerance is satisfied')
                    p_k = z_j1
                    assert np.linalg.norm(p_k) <= delta
                    return p_k, 1
                else:
                    #print('Compute new direction')
                    beta_j1 = (r_j1**2).sum()/(r_j**2).sum()
                    d_j1 = -r_j1 + beta_j1*d_j
                    return z_j1, d_j1, r_j1, 0

    def inner_loop(self, x_k, delta):
        tol = np.minimum(0.3, np.linalg.norm(self.obj.delta_f(x_k))**(1/2))*np.linalg.norm(self.obj.delta_f(x_k))
        z_old = np.zeros(self.N)
        d_old = - self.obj.delta_f(x_k)
        r_old = self.obj.delta_f(x_k)

        iteration = 1
        crit = 0
        while (crit ==0) & (iteration < 1000):
            out = self.step_inner_loop_CG(z_old, d_old, r_old, x_k, delta, tol)
            if out[-1]==1:
                p_k = out[0]
                crit = out[-1]
                return p_k, iteration
            else:
                z_old, d_old, r_old = out[:-1]

            iteration += 1

    def outer_loop(self, x_0, delta_0):
        tol = 1e-5
        crit = 1e5

        x_k = x_0
        delta_k = delta_0

        it = 1
        while crit > tol and it < 1000:
            #print(crit, it, delta_k)
            p_k, cg_iter = self.inner_loop(x_k, delta_k)
            x_k1, delta_k1 = self.update_trust_region(x_k, p_k, delta_k)

            tol_cg = np.minimum(0.3, np.linalg.norm(self.obj.delta_f(x_k))**(1/2))*np.linalg.norm(self.obj.delta_f(x_k))
            _print_each_iter(it, self.obj.f(x_k1), np.linalg.norm(self.obj.delta_f(x_k1)), tol_cg,  cg_iter)
            crit = np.linalg.norm(self.obj.delta_f(x_k1))
            x_k = x_k1
            delta_k = delta_k1

            it += 1

        return x_k, self.obj.delta_f(x_k)


def _print_first_iter():
    print('iter '
         + 'f            '
         + '||grad_f||   '
         + 'CG tol  '
         + 'CG iterations' )

    pass

def _print_each_iter(i, obj,  grad_norm, epsilon, cg_iter):

    print('%i   %4.04e   %4.04e   %4.04f      %i' % (i, obj,  grad_norm, epsilon, cg_iter))

    pass

N = 1000
delta_hat = 100
delta_0 = 1
eta = 0.1
x_0 = np.ones(N)

tr_instance = TrustRegion(N, delta_hat, eta)

print('Trust-Region CG')
print('=========================================================')
_print_first_iter()
print('----------------------------------------------------------')

x_star, delta_x_star = tr_instance.outer_loop(x_0, delta_0)

#p, it = tr_instance.inner_loop(x_0, delta_0)