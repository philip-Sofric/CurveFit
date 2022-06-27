import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import matplotlib.pyplot as plt


def WhittakerSmooth(x, w, lambda_, differences=1):
    """
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    """
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    """
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    """
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * (abs(x)).sum() or i == itermax:
            if i == itermax: print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


def LSE(x1, x2):
    length = len(x1)
    ss = 0
    for i in range(length):
        ss += (x1[i] - x2[i]) ** 2
    return np.sqrt(ss)


# First derivative of vector using 2-point central difference.
def deriv(yarray):
    length = len(yarray)
    der = np.zeros(length)
    for i in range(length):
        if i == 0:
            der[i] = yarray[i+1] - yarray[i]
        elif i == length - 1:
            der[i] = yarray[i] - yarray[i-1]
        else:
            der[i] = (yarray[i+1] - yarray[i-1]) / 2
    return der


xx = np.arange(0, 1000, 1)
g1 = norm(loc=100, scale=1.0)  # generate three gaussian as a signal
g2 = norm(loc=300, scale=3.0)
g3 = norm(loc=750, scale=15.0)
peak1 = g1.pdf(xx) * 1
peak2 = g2.pdf(xx) * 1
peak3 = g3.pdf(xx) * 1
signal = peak1 + peak2 + peak3
# baseline1 = 5e-4 * x + 0.2  # linear baseline
baseline2 = 0.2 * np.sin(np.pi * xx / xx.max())  # sinusoidal baseline
baseline = baseline2
noise = np.random.random(xx.shape[0]) / 500
y = signal + baseline + noise
plt.plot(xx, y, 'k', label='Raw spectrum')
opt_lb = 42
# opt_lse = LSE(baseline, airPLS(y, 1))
# for lb in range(100):
#     tt = LSE(baseline, airPLS(y, lb))
#     if tt < opt_lse:
#         opt_lb = lb
#         opt_lse = tt
# print('Optimized lambda is ', opt_lb)
baseline_calc = airPLS(y, opt_lb)
y_corrected = y - baseline_calc
y_derivative = deriv(y)
y_deriv_corr = deriv(y_corrected)
# plt.plot(xx, baseline, 'c', label='Original Baseline')
# plt.plot(xx, baseline_calc, 'g', label='Calculated Baseline')
plt.plot(xx, y_derivative, 'r', label='1st derivative')
plt.plot(xx, y_deriv_corr, 'c', label='1st derivative on corrected')
# plt.plot(xx, y_corrected, 'k', label='Baseline removed spectrum')

plt.title('Baseline Removal')
plt.legend()
plt.show()
print('Done!')
