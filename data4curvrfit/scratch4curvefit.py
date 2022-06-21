import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy import interpolate


def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)
    y = a + b * np.exp(t * c)
    error = noise * rng.standard_normal(t.size)
    outliers = rng.integers(0, t.size, n_outliers)
    error[outliers] *= 10
    return y + error


def fun(x, t, y):
    return x[0] + x[1] * np.exp(x[2] * t) - y


a = 0.5
b = 2.0
c = -1
t_min = 0
t_max = 10
n_points = 15
x0 = np.array([1.0, 1.0, 0.0])

t_train = np.linspace(t_min, t_max, n_points)
y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)
res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
                            args=(t_train, y_train))
t_test = np.linspace(t_min, t_max, n_points * 10)
y_test = gen_data(t_test, a, b, c)
tck = interpolate.splrep(t_train, y_train, s=0)
y_interp = interpolate.splev(t_test, tck, der=0)

plt.plot(t_test, y_interp, label='interpolation')
plt.plot(t_train, y_train, label='before fit')
plt.plot(t_test, y_test, label='fitted')

plt.legend()
plt.show()