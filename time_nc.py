#%%
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt

path = '/home/intern/data/CMAQ/CCTM/CCTM_ACONC_v54_intel_D02_2021_20210402.nc'
data = Dataset(path, mode = 'r')

#TFLAG(시간)//O, NO, NO2  
TFLAG = np.squeeze(data.variables['TFLAG'][:, 0, :])
times = TFLAG[:, 1]  # 시간 정보
hhmmss = [str(int(time)).zfill(6)[:-4] for time in times]

plt.figure(figsize=(10, 6))
# 첫번째 y축
O = np.squeeze(data.variables['O'][:, 0, 0, 0])
plt.plot(hhmmss, O, label='O [Pa]', marker='o', color='r')
plt.ylabel('[Pa]', color='k')
plt.tick_params(axis='y', labelcolor='k')
plt.legend(loc='upper left')


# 두 번째 y축 (NO, NO2 데이터)
ax2 = plt.twinx()
NO = np.squeeze(data.variables['NO'][:, 0, 0, 0])
NO2 = np.squeeze(data.variables['NO2'][:, 0, 0, 0])
ax2.plot(hhmmss, NO, label='NO [ppmV]', marker='o', color='b')
ax2.plot(hhmmss, NO2, label='NO2 [ppmV]', marker='o', color='g')
ax2.set_ylabel('[ppmV]', color='k')
#ax2.tick_params(axis='y', labelcolor='k')

plt.legend(loc='upper right')


plt.title('O, NO, and NO2 Concentrations 20210401')
plt.xlabel('Time (HOUR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




#%%
variables = ['O', 'NO', 'NO2']
labels = ['O [Pa]', 'NO [ppmV]', 'NO2 [ppmV]']

for var, label in zip(variables, labels):
    var = np.squeeze(data.variables[var][:,0,0,0])
    plt.plot(hhmmss, var, label=label, marker='o')
#%%
