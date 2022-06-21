import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy import interpolate
from scipy.optimize import least_squares

# find initial peak parameters: height, center and standard deviation - sigma; sqrt(2ln2)*sigma = FWHM/2
def find_initials(peak_position, xdata, ydata):


# height coef[0]; center coef[1]; standard deviation coef[2]
def Gaussian(coef, xdata):
    return coef[0] * np.exp(-((xdata - coef[1]) / coef[2]) ** 2 / 2)


def fun(coef, xdata, ydata):
    n_peak = len(coef) // 3
    sum_gaussian = 0
    for i in np.arange(n_peak):
        coef1 = [coef[i*3], coef[i*3+1], coef[i*3+2]]
        sum_gaussian += Gaussian(coef1, xdata)
    return sum_gaussian - ydata


path1 = r'D:\Projects\Algorithm Study\CurveFit'
path2 = r'D:\Projects\Algorithm Study\CurveFit\data4curvrfit'

os.chdir(path2)
df1 = pd.read_csv('sample0.csv')
Column1 = df1['RamanShift']
Column2 = df1['Processed']
RS_interp = np.arange(150, 2500, 10)
tck = interpolate.splrep(Column1, Column2, s=0)
intensity_interp = interpolate.splev(RS_interp, tck, der=0)
# plt.plot(Column1, Column2, label='original')
plt.plot(RS_interp, intensity_interp, label='interpolation')

coeff = [20000, 882, 10, 5000, 1450, 20]
# res_soft_l1 = least_squares(fun, coeff, method='lm',
#                             args=(Column1, Column2))
res_lm = least_squares(fun, coeff, method='lm',
                       args=(RS_interp, intensity_interp))
new_coeff = res_lm.x
print(f'New coefficients are ', new_coeff)
print(f'cost is ', res_lm.cost)
num_peak = len(new_coeff) // 3
for j in np.arange(num_peak):
    y_lm = Gaussian(new_coeff[3*j:3*(j+1)], RS_interp)
    labels = 'lm' + str(j)
    plt.plot(RS_interp, y_lm, label=labels)
plt.legend()
plt.show()
