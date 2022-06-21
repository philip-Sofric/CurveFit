import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import derivative

path1 = r'D:\Projects\Algorithm Study\CurveFit'
path2 = r'D:\Projects\Algorithm Study\CurveFit\data4curvrfit'

os.chdir(path2)
df1 = pd.read_csv('sample0.csv')
Column1 = df1['RamanShift']
Column2 = df1['Processed']
# plt.plot(Column1, Column2)

Index_0 = np.where(Column2 == max(Column2))
xx = Column1[366]

RS_interp = np.arange(150, 2500, 10)
print(xx)
# plt.show()
