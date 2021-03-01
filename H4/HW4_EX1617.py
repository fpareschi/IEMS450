import numpy as np
import matplotlib.pyplot as plt


def f(x_1, x_2):
    return 6*x_1 + 4*x_2 - 13 - x_1**2 - x_2**2

delta = 0.01
x1 = np.arange(-3, 4, delta)
x2 = np.arange(-3, 4, delta)

X1, X2 = np.meshgrid(x1, x2)

Z = f(X1, X2)

fig, ax = plt.subplots()
ax.scatter(0, 0, color='black')
CS = ax.contour(X1, X2, Z, levels=70)
ax.set_title('Graphical Solution')
ax.fill_between(x1[(x1>0) & (x1<3)], 0, 3-x1[(x1>0) & (x1<3)], facecolor='g', alpha=0.4)
#ax.fill_between(x1, 0, 1000, facecolor='g', alpha=0.4)
#ax.axvspan(0, 1000, alpha=0.4,  color='green')
#ax.axvspan(0, 1000, alpha=0.3,  color='green')
ax.set_xlim([-1, 4])
ax.set_ylim([-1, 4])
plt.show()