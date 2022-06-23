import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def Gaussian(coef, xdata):
    return coef[0] * np.exp(-((xdata - coef[1]) / coef[2]) ** 2 / 2)


def fun(coef, xdata, ydata):
    n_peak = len(coef) // 3
    sum_gaussian = 0
    for i in np.arange(n_peak):
        coef1 = [coef[i * 3], coef[i * 3 + 1], coef[i * 3 + 2]]
        sum_gaussian += Gaussian(coef1, xdata)
    return sum_gaussian - ydata


x = np.arange(0, 1000, 1)
# generate three gaussian as a signal
g1 = norm(loc=400, scale=50.0)
g2 = norm(loc=500, scale=25.0)
# print(type(g2))
# g3 = norm(loc=750, scale=5.0)
signal1 = g1.pdf(x) * 50
signal2 = g2.pdf(x) * 8
signal = signal1 + signal2
baseline = 0.2 * np.sin(np.pi * x / x.max())  # sinusoidal baseline
noise = np.random.random(x.shape[0]) / 500
y = signal + baseline + noise
plt.plot(x, y, '-r', label='Raw spectrum')
plt.plot(x, signal1, 'k', label='peak 1')

# a0 = 1/np.sqrt(2*np.pi)
# RS_interp = np.arange(-5, 5, 0.1)
# coeff = [a0, 0, 1]
# y_Gaussian = Gaussian(coeff[:3], RS_interp)
# plt.grid(visible=True)
# plt.ylim(0, 1)
# plt.plot(RS_interp, y_Gaussian, label='Gaussian')
# y_Gaussian2 = Gaussian(coeff[3:6], RS_interp)
# plt.plot(RS_interp, y_Gaussian2, label='Gaussian2')
# plt.plot(RS_interp, y_Gaussian + y_Gaussian2, label='Gaussian2')
plt.legend()
plt.show()
