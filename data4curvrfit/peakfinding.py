import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter as sg


def peak_slope(peak_range_in, xarray, yarray):
    index_peak_start = np.where(xarray == peak_range_in[0])
    index_peak_end = np.where(xarray == peak_range_in[1])
    intensity_peak_start = yarray[index_peak_start]
    intensity_peak_end = yarray[index_peak_end]
    slope = (intensity_peak_end - intensity_peak_start) / (peak_range_in[1] - peak_range_in[0])
    intercept = intensity_peak_start - slope * peak_range_in[0]
    return slope * xarray + intercept


# First derivative of vector using 2-point central difference.
def deriv(darray):
    length = len(darray)
    der = np.zeros(length)
    for i in range(length):
        if i == 0:
            der[i] = darray[i + 1] - darray[i]
        elif i == length - 1:
            der[i] = darray[i] - darray[i - 1]
        else:
            der[i] = (darray[i + 1] - darray[i - 1]) / 2
    return der


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


def peakfind(darray, slopethresh=0.001, intensthresh=0.1):
    deri_array = deriv(darray)
    smoothed_derivative = sg(deri_array, 51, 3)
    length = len(darray)
    for i in range(length):
        if smoothed_derivative[i] >= slopethresh:
            continue
    return


x = np.arange(0, 1000, 1)
g1 = norm(loc=400, scale=50.0)  # generate three gaussian as a signal
g2 = norm(loc=500, scale=25.0)
# g3 = norm(loc=750, scale=5.0)
signal1 = g1.pdf(x) * 50
signal2 = g2.pdf(x) * 8
signal = signal1 + signal2
baseline1 = 5e-4 * x + 0.2  # linear baseline
baseline2 = 0.2 * np.sin(np.pi * x / x.max())  # sinusoidal baseline
baseline = baseline2
noise = np.random.random(x.shape[0]) / 500
y = signal + baseline + noise
plt.plot(x, y, 'r', label='original')

# baseline removal
baseline_calc = airPLS(y, 10)
y2 = y - baseline_calc
# plt.plot(x, baseline_calc, 'y', label='baseline')
# plt.plot(x, y2, 'k', label='corrected')
# peak finding
y_deriv = deriv(y)
y2_deriv = deriv(y2)
plt.plot(x, y_deriv, 'c', label='1st derivative on original')
# plt.plot(x, y2_deriv, 'r', label='derivative on corrected')
y2_deriv_smooth = sg(y2_deriv, 51, 3)
plt.plot(x, y2_deriv_smooth, 'k', label='smoothed derivative on original')


# path1 = r'D:\Projects\Algorithm Study\CurveFit'
# path2 = r'D:\Projects\Algorithm Study\CurveFit\data4curvrfit'
#
# os.chdir(path2)
# df1 = pd.read_csv('sample0.csv')
# Column1 = df1['RamanShift']
# Column2 = df1['Processed']
# # plt.plot(Column1, Column2, label='original')
# RS_interp = np.arange(150, 2500, 1)
# tck = interpolate.splrep(Column1, Column2, s=0)
# intensity_interp = interpolate.splev(RS_interp, tck, der=0)
# plt.plot(RS_interp, intensity_interp, label='interpolation')
#
# peak_range = [770, 960]
# y2 = peak_slope(peak_range, RS_interp, intensity_interp)
# plt.plot(RS_interp, intensity_interp)
# # plt.fill_between(RS_interp, intensity_interp,
# #                  where=(RS_interp >= peak_range[0]) & (RS_interp <= peak_range[1]),
# #                  color='orange')
# plt.fill_between(RS_interp, intensity_interp, y2,
#                  where=(RS_interp >= peak_range[0]) & (RS_interp <= peak_range[1]),
#                  color='green')

plt.legend()
plt.show()
