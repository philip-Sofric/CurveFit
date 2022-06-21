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
# plt.plot(RS_interp, intensity_interp, label='interpolation')

peak_range = [770, 960]
y2 = peak_slope(peak_range, RS_interp, intensity_interp)
plt.plot(RS_interp, intensity_interp)
# plt.fill_between(RS_interp, intensity_interp,
#                  where=(RS_interp >= peak_range[0]) & (RS_interp <= peak_range[1]),
#                  color='orange')
plt.fill_between(RS_interp, intensity_interp, y2,
                 where=(RS_interp >= peak_range[0]) & (RS_interp <= peak_range[1]),
                 color='green')
print()
# plt.legend()
plt.show()
