import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import interpolate


def peak_slope(peak_range_in, xarray, yarray):
    index_peak_start = np.where(xarray == peak_range_in[0])
    index_peak_end = np.where(xarray == peak_range_in[1])
    intensity_peak_start = yarray[index_peak_start]
    intensity_peak_end = yarray[index_peak_end]
    slope = (intensity_peak_end - intensity_peak_start) / (peak_range_in[1] - peak_range_in[0])
    intercept = intensity_peak_start - slope * peak_range_in[0]
    return slope * xarray + intercept

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


def findpeak(darray, slopethresh, intensitythresh):
    length = len(darray)
    darray_der = deriv(darray)
    epsilon = 0.0001
    peak_array = []
    if length < 5:
        print('There has to be no less than 5 data points not enough to form a peak! ')
        return -1
    else:
        i = 0
        while i < (length - 1):
            if darray_der[i] > darray_der[i+1] and darray_der[i] >= slopethresh and darray[i] >= intensitythresh:
                for j in np.arange(i+1, length, 1):
                    if darray_der[j] > epsilon:
                        continue
                    elif abs(darray_der[j]) <= epsilon:
                        peak_array.append(j)
                        i = j + 1
                        break
                    else:  # darray_der[j] < -epsilon
                        peak_array.append(j)
                        i = j + 1
                        break
            else:
                i += 1
    return peak_array

def peak_info(darray, peak_index):


# path1 = r'D:\Projects\Algorithm Study\CurveFit'
# path2 = r'D:\Projects\Algorithm Study\CurveFit\data4curvrfit'
path1 = r'/Users/philip/PycharmProjects/CurveFit'
path2 = r'/Users/philip/PycharmProjects/CurveFit/data4curvrfit'

os.chdir(path2)
df1 = pd.read_csv('smooth on baseline corrected.csv')
x = df1['x']
y = df1['y']
plt.plot(x,y,'c',label='smooth on baseline corrected.csv')
plt.plot(x, deriv(y), 'r', label='derivative of smoothed')
peaks = findpeak(y, 0.001, 0.001)
print(peaks)


# Column1 = df1['RamanShift']
# Column2 = df1['Processed']
# # plt.plot(Column1, Column2, label='original')
# RS_interp = np.arange(150, 2500, 1)
# tck = interpolate.splrep(Column1, Column2, s=0)
# intensity_interp = interpolate.splev(RS_interp, tck, der=0)
# # plt.plot(RS_interp, intensity_interp, label='interpolation')
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
print()
plt.legend()
plt.show()
