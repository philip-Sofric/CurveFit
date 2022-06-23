import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from scipy import interpolate
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter


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


def airPLS(x_, lambda_=100, porder=1, itermax=15):
    """
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x_: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    """
    m = x_.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x_, w, lambda_, porder)
        d = x_ - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * (abs(x_)).sum() or i == itermax:
            if i == itermax:
                print('WARING max iteration reached!')
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
        ss += (x1[i] - x2[i])**2
    return np.sqrt(ss)


x = np.arange(0, 1000, 1)
g1 = norm(loc=400, scale=50.0)  # generate three gaussian as a signal
g2 = norm(loc=500, scale=25.0)
g3 = norm(loc=750, scale=5.0)
signal = g1.pdf(x) + g2.pdf(x) + g3.pdf(x)
baseline1 = 5e-4 * x + 0.2  # linear baseline
baseline2 = 0.2 * np.sin(np.pi * x / x.max())  # sinusoidal baseline
baseline = baseline2
noise = np.random.random(x.shape[0]) / 500
y = signal + baseline + noise
# opt_lb = 1
# opt_diff = LSE(baseline, airPLS(y, 1))
# for lb in range(100):
#     diff = LSE(baseline, airPLS(y, lb))
#     if diff < opt_diff:
#         opt_lb = lb
#         opt_diff = diff
# print('Optimized lambda for baseline removal is ', opt_lb)
# baseline_calc = airPLS(y, opt_lb)
# y_2 = y - baseline_calc  # with baseline removed
# plt.plot(x, y, '-k', label='original')
# # plt.plot(x, y_2, '-r', label='corrected')
# plt.plot(x, baseline, '-b', label='original baseline')
# plt.plot(x, baseline_calc, '.r', label='calculated baseline')
# plt.title('Sinusoidal baseline')

path1 = r'D:\Projects\Algorithm Study\CurveFit'
path2 = r'D:\Projects\Algorithm Study\CurveFit\data4curvrfit'

os.chdir(path2)
df1 = pd.read_csv('sample0.csv')
Column1 = df1['RamanShift']
Column2 = df1['Processed']
# plt.plot(Column1, Column2, label='original')
RS_interp = np.arange(150, 2500, 1)
tck = interpolate.splrep(Column1, Column2, s=0)
intensity_interp = interpolate.splev(RS_interp, tck, der=0)
plt.plot(RS_interp, intensity_interp, label='interpolation')
baseline = airPLS(intensity_interp, 50)
c3 = intensity_interp - baseline
plt.plot(RS_interp, baseline, label='Baseline')
plt.plot(RS_interp, c3, label='Corrected')

plt.legend()
plt.show()
print('Done!')
