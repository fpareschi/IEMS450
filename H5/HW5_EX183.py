import numpy as np


class OptProgram(object):

    def __init__(self, n):
        self.n = n

    def f(self, x):
        assert x.shape[0] == self.n
        f = np.exp(np.prod(x)) - 0.5 * ( x[0]**3 + x[1]**3 + 1 )**2
        return f

    def df(self,x):
        delta_f = np.zeros(x.shape[0])

        indexes = [i for i in range(x.shape[0])]
        indexes = set(indexes)

        for i, val in enumerate(x):
            delta_f[i] = (np.exp(x.prod()))*(x[list(indexes-{i})].prod())

        #delta_f[0] += -(x[0]**3 + x[1]**3 + 1)*(3*x[0]**2)
        #delta_f[1] += -(x[0] ** 3 + x[1] ** 3 + 1) * (3 * x[1] ** 2)

        return delta_f

    def ddf(self, x):
        indexes = [i for i in range(x.shape[0])]
        indexes = set(indexes)

        ddelta_f = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if i == j:
                    ddelta_f[i,j] = np.exp(x.prod())*((x[list(indexes-{i})].prod())**2)

                elif i>j:
                    ddelta_f[i, j] = np.exp(x.prod())*x[list(indexes - {i})].prod()*x[list(indexes - {j})].prod() + \
                                     np.exp(x.prod())*x[list(indexes - {j,i})].prod()


        #ddelta_f[0,0] += -9*x[0]**4 - 6*x[0]*(x[0]**3 + x[1]**3 + 1)
        #ddelta_f[1,0] += (-3*(x[0]**2))*(-3*(x[1]**2))

        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if i<j:
                    ddelta_f[i,j] = ddelta_f[j,i]

        assert np.allclose(ddelta_f, ddelta_f, rtol=1e-6, atol=1e-6)
        return ddelta_f

    def c1(self, x):
        return (x**2).sum() - 10

    def dc1(self, x):
        return 2*x

    def ddc1(self, x):
        return 2*np.eye(x.shape[0])

    def c2(self,x):
        return x[1]*x[2] - 5*x[3]*x[4]

    def dc2(self, x):
        return np.array([0, x[2], x[1], -5*x[4], -5*x[3]])

    def ddc2(self, x):
        return np.array([[0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, -5],
                        [0, 0, 0, -5, 0]])

    def c3(self, x):
        return x[0]**3 + x[1]**3 + 1

    def dc3(self, x):
        return np.array([3*(x[0]**2), 3*(x[1]**2), 0, 0, 0])

    def ddc3(self, x):
        return np.array([[6*x[0], 0, 0, 0, 0],
                        [0, 6*x[1], 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    def dlag(self, x, lambda_k):
        return self.df(x) - lambda_k[0]*self.ddc1(x) - lambda_k[1]*self.ddc2(x) - lambda_k[2]*self.ddc3(x)

n = 5
x0 = np.array([-1.8, 1.7, 1.9, -0.8, -0.8])
lambda_0 = np.array([1,1,1])
fcts = OptProgram(n)

def solve_directions(x_k, lambda_k):

    L2 = fcts.ddf(x_k) - lambda_k[0]*fcts.ddc1(x_k) - lambda_k[1]*fcts.ddc2(x_k) - lambda_k[2]*fcts.ddc3(x_k)
    assert L2.shape == (n, n)

    Ak = np.vstack([fcts.dc1(x_k)[np.newaxis, :],fcts.dc2(x_k)[np.newaxis, :], fcts.dc3(x_k)[np.newaxis, :] ])
    assert Ak.shape == (3, n)

    up_block = np.hstack([L2 , -Ak.T])
    lw_block = np.hstack([Ak, np.zeros((3,3))])

    LHS = np.vstack([up_block, lw_block])
    RHS = np.array([*fcts.df(x0) * (-1), -fcts.c1(x0), -fcts.c2(x0), -fcts.c3(x0)])
    p = np.linalg.solve(LHS, RHS)
    x_k1 = x_k + p[:-3]
    lambda_k1 = p[-3:]
    return x_k1, lambda_k1

tol = 1e-6
crit = 100
it  = 0
x_k = x0
lambda_k = lambda_0
while (crit > tol) & (it<100):

      x_k, lambda_k = solve_directions(x_k, lambda_k)
      crit = np.linalg.norm(fcts.dlag(x_k, lambda_k))
      it += 1
      print(crit, it)