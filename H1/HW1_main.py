import numpy as np

np.random.seed(30)

m, L = 0.01, 1
D = 10 ** np.random.rand(100)
D = (D - np.min(D)) / (np.max(D) - np.min(D))
A = np.diag(m + D * (L - m))

# Starting Point
S = 10
x_initial = np.random.rand(100, S)


class LinearSearch(object):

    def __init__(self, A, m, L):
        self.A = A
        self.m = m
        self.L = L


    def f(self, x):
        return 0.5 * (x.T @ self.A @ x)


    def delta_f(self, x):
        return x.T @ self.A

    def steepest_descend(self, x_k, alpha):
        x_k1 = x_k - self.delta_f(x_k)*alpha
        return x_k1

    def heavy_ball(self, x_k, x_k_1):
        beta = (np.sqrt(self.L) - np.sqrt(self.m))/(np.sqrt(self.L) + np.sqrt(self.m))
        alpha = 4/((np.sqrt(self.L) + np.sqrt(self.m))**2)
        x_k1 = x_k - alpha*self.delta_f(x_k) + beta*(x_k - x_k_1)
        return x_k1

    def nesterov(self, x_k, x_k_1, t_k):
        t_k1 = 0.5*(1 + np.sqrt(1 + 4*(t_k**2)))
        y_k1 = x_k + ((t_k - 1)/(t_k1))*(x_k - x_k_1)
        x_k1 = y_k1 - (1/self.L)*self.delta_f(y_k1)
        return x_k1, t_k1


def linear_search(LS_inst, x_k, x_k_1, t_k, method='exact_steepest_descend'):

    if method=='exact_steepest_descend':
        print(LS_inst.delta_f(x_k).shape, LS_inst.A.shape)
        alpha_ = (LS_inst.delta_f(x_k).T @ LS_inst.delta_f(x_k))/(LS_inst.delta_f(x_k).T @ LS_inst.A @ LS_inst.delta_f(x_k))
        x_k1 = LS_inst.steepest_descend(x_k, alpha_)
        return x_k1
    elif method=='stepping_steepest_descend':
        x_k1 = LS_inst.steepest_descend(x_k, 1/LS_inst.L)
        return x_k1
    elif method =='heavy_ball':
        x_k1 = LS_inst.heavy_ball(x_k, x_k_1)
        return x_k1
    elif method == 'nesterov':
        '''In this case the next step is on y_k1. Keep name for return only.'''
        x_k, t_k1 = LS_inst.nesterov(x_k, x_k_1, t_k)
        return x_k, t_k1

    else:
        raise ValueError('This linear search method is not implemented')


tol = 1e-6
main_instance = LinearSearch(A, m, L)
methods = ['stepping_steepest_descend', 'exact_steepest_descend', 'heavy_ball', 'nesterov']
iteration_count = np.zeros((S, len(methods)))
log_dict = {}
for s in range(S):

    for ll, method in enumerate(methods):
        x_0 = x_initial[:, s]
        x_1 = main_instance.steepest_descend(x_0, 1 / L)

        t_0 = 1
        crit = 1000
        log_dict[method, s] = []
        i = 0

        while crit>tol:

            next_step = linear_search(main_instance, x_0, x_1, t_0, method=method)
            if method == 'nesterov':
                x1 = next_step[0]
                t1 = next_step[1]
                t_0 = t1
            else:
                x1 = next_step

            crit = main_instance.f(x1) - main_instance.f(np.zeros(100))

            x_1 = x_0
            x_0 = x1
            i +=1
            log_dict[method, s].append(np.log(crit))

        print(method)
        iteration_count[s, ll] = i

import matplotlib.pyplot as plt

colors = ['red', 'blue', 'black', 'green']
fig, ax = plt.subplots(2,1)
for i in range(4):
    ax[0].plot(range(len(log_dict[methods[i], 0])), np.array(log_dict[methods[i], 0]), label=methods[i], color=colors[i])
ax[0].legend()

for i in range(4):
    ax[1].plot(range(len(log_dict[methods[i], 0])), np.array(log_dict[methods[i], 0]), label=methods[i], color=colors[i],
               linewidth=0.2)
    for j in range(S):
        ax[1].plot(range(len(log_dict[methods[i], s])), np.array(log_dict[methods[i], s]), color=colors[i], linewidth=0.2)

ax[1].legend()
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('$log(f(x_k) - f(x^*))$')

ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('$log(f(x_k) - f(x^*))$')

plt.show()

file = open('C:/Users/franc/Google Drive/Northwestern/Winter - 2021/Optimization - 450-2/HW/HW1/Tab1.tex', 'w')
file.write('\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcc')
file.write('} \n ')
file.write('\\toprule \n')
file.write(' & Mean Iterations & Std. Iterations \\\\ \n')
file.write('\\midrule \n')
for i in range(4):
    file.write('%s & %4.02f & %4.02f \\\\ \n' % (methods[i], iteration_count[:, i].mean(), iteration_count[:, i].std()))
file.write('\\bottomrule \n')
file.write('\\end{tabular*} \n')
file.close()
