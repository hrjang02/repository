#%%
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import pyresample
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

year = '2021'
'''
Observation의 data 불러오기(시간 : KST) -> 자정(00:00)의 시간만 가져오기
'''

file= pd.read_csv(f'/home/intern/data/pollutant/{year}_PM25.csv')
# OBS의 station,lon, lat
obs_stn = file.columns[2:].str.split('_').str[0]
obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)

file = file.loc[f'{year}-06-03':f'{year}-06-04']
file = file[file.index.hour == 0]
file = file.drop(columns='Unnamed: 0')


'''
CMAQ data -> 각 날짜에 대해 자정(00:00) 데이터 처리
'''

path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{year}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values


merge_list = []
#각 날짜별 KST의 0시의 데이터를 가져옴
for date in pd.date_range(start=f'{year}-06-03', end=f'{year}-06-04', freq='D'):
    path_cmaq_o3 = f'/home/intern/data/CMAQ/CCTM/CCTM_ACONC_v54_intel_D02_{year}_{date.strftime("%Y%m%d")}.nc'
    cmaq_o3 = Dataset(path_cmaq_o3, mode='r')
    O3 = np.squeeze(cmaq_o3.variables['O3'][15, 0, :, :])  # 00시 데이터만 선택(+9시간)
    
    obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
    CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

    wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

    O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                               radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                               fill_value=None)
    
    
    
    obs_date_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    obs_data = file.loc[obs_date_str]    
    
    CMAQ_df = pd.DataFrame({'datetime':obs_date_str, 'station': obs_stn, 'lon': obs_lon, 'lat': obs_lat, 'CMAQ_O3': O3_re, 'OBS_O3': obs_data})
    merge_list.append(CMAQ_df)

Night_o3= pd.concat(merge_list, ignore_index=True)

#%%
#Night_o3['datetime'] = pd.to_datetime(Night_o3['datetime'])
file_monthly = Night_o3.resample('D').mean()
#Night_o3.set_index('datetime', inplace=True)
#
#%%

#row_means = file_monthly.mean(axis=0)


#%%
'''
오차 상관계수 구해서 그림 그리기
'''

obs_data_orig = Night_o3.iloc[:, 5].values
cmaq_data_orig = Night_o3.iloc[:, 4].values

# NaN 값이 있는거 제외
valid_mask = ~np.isnan(obs_data_orig) & ~np.isnan(cmaq_data_orig)
obs_data= obs_data_orig[valid_mask]
cmaq_data = cmaq_data_orig[valid_mask]

# 상관 계수 계산
correlation_matrix = np.corrcoef(obs_data, cmaq_data)
correlation_coefficient = correlation_matrix[0, 1]
print(f'상관 계수 (R): {correlation_coefficient}')

# 상관 계수 계산
def calculator(actual, predicted):
    ME2 = np.abs(predicted-actual)
    ME = np.mean(ME2)
    NME = (np.sum(ME2)/np.sum(actual))*100
    MB = np.mean(predicted-actual)
    NMB = (np.sum(predicted-actual)/np.sum(actual))*100
    RMSE = np.sqrt(np.mean((predicted-actual)**2))
    
    corr,_ = pearsonr(actual, predicted)
    corr = format(corr,"5.2f")

    ME = format(ME,"5.2f")
    NME = format(NME,"5.2f")
    MB = format(MB,"5.2f")
    NMB = format(NMB,"5.2f")
    RMSE = format(RMSE,"5.2f")
    return corr,MB,ME,NMB,NME,RMSE

print(calculator(obs_data, cmaq_data))


slope, intercept = np.polyfit(obs_data, cmaq_data, 1)
regression_line = slope * obs_data + intercept


fig, ax = plt.subplots(figsize=(12, 10))
hb = ax.hexbin(obs_data, cmaq_data, gridsize=50, cmap='Reds', mincnt=1)
ax.set_xlabel('Observed O3 Concentration (ppm)')
ax.set_ylabel('CMAQ Interpolated O3 Concentration (ppm)')
ax.set_title(f'Validation of Observed and CMAQ O3 Concentrations 4-6,{year}')
fig.colorbar(hb, ax=ax, label='Counts')


ax.plot(obs_data, regression_line, 'r-', label=f'Regression line\ny = {slope:.2f}x + {intercept:.2f}')
ax.plot([min(obs_data), max(obs_data)], [min(obs_data), max(obs_data)], 'k--', linewidth=2, label='1:1 Line')

ax.set_xlim([0, 0.10])  
ax.set_ylim([0, 0.10])  

ax.grid(True)
ax.legend()
plt.savefig(f'/home/hrjang/AICP/figure/Validation of Observed and CMAQ O3 Concentrations 4-6,{year}.png')
# %%
