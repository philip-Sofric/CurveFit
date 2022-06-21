import numpy as np
import matplotlib.pyplot as plt


def Gaussian(coef, xdata):
    return coef[0] * np.exp(-((xdata - coef[1]) / coef[2]) ** 2 / 2)


def fun(coef, xdata, ydata):
    n_peak = len(coef) // 3
    sum_gaussian = 0
    for i in np.arange(n_peak):
        coef1 = [coef[i*3], coef[i*3+1], coef[i*3+2]]
        sum_gaussian += Gaussian(coef1, xdata)
    return sum_gaussian - ydata


a0 = 1/np.sqrt(2*np.pi)
RS_interp = np.arange(-5, 5, 0.1)
coeff = [a0, 0, 1]
y_Gaussian = Gaussian(coeff[:3], RS_interp)
plt.grid(visible=True)
plt.ylim(0, 1)
plt.plot(RS_interp, y_Gaussian, label='Gaussian')
# y_Gaussian2 = Gaussian(coeff[3:6], RS_interp)
# plt.plot(RS_interp, y_Gaussian2, label='Gaussian2')
# plt.plot(RS_interp, y_Gaussian + y_Gaussian2, label='Gaussian2')
plt.show()
