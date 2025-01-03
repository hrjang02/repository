#%%
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import pyresample
from scipy.interpolate import griddata
import xarray as xr

# Observation data + 위경도 정보
year = '2021'
var = 'NO2'
csv_directory = '/home/aicp/aicp419/data/pollutant'
df_obs = pd.read_csv(f'{csv_directory}/{year}_{var}.csv')
df_obs['측정일시'] = pd.to_datetime(df_obs['측정일시'])

obs_stn = df_obs.columns[2:].str.split('_').str[0]
obs_lon = df_obs.columns[2:].str.split('_').str[1].astype(float)
obs_lat = df_obs.columns[2:].str.split('_').str[2].astype(float)

df_obs.set_index('측정일시', inplace=True)
header_str = list(obs_stn)
filtered_columns = [col for stn_id in header_str for col in df_obs.columns if col.startswith(str(stn_id))]
data_obs = df_obs[filtered_columns]

data_obs = data_obs.loc['2021-04-01':'2021-04-01']
data_obs.index = data_obs.index.tz_localize('Asia/Seoul')
data_obs.index = data_obs.index.tz_convert('UTC')

obs_data = data_obs.iloc[0].values

#%%
# CMAQ data + 위경도 정보 -------------------------------------------------------------
path_cmaq_no2 = '/home/aicp/aicp419/data/CMAQ/CCTM/CCTM_ACONC_v54_intel_D02_2021_20210401.nc'
data_no2 = Dataset(path_cmaq_no2, mode='r')
NO2 = np.squeeze(data_no2.variables['NO2'][0, 0, :, :])

path_cmaq = xr.open_dataset(f'/home/aicp/aicp419/data/CMAQ/MCIP/GRIDCRO2D_D02_2021.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values

#%%
# CMAQ 격자 좌표의 평탄화
predict_lon_flat = lon_CMAQ.ravel()  # 평탄화된 경도
predict_lat_flat = lat_CMAQ.ravel()  # 평탄화된 위도
NO2_flat = NO2.ravel() 

predict_matching_station = griddata((predict_lon_flat, predict_lat_flat),NO2_flat,(obs_lon, obs_lat),
                                    method='cubic')


# NaN 값이 있는 행 제외하고 상관 계수 계산
valid_mask = ~np.isnan(obs_data) & ~np.isnan(predict_matching_station)
obs_data_valid = obs_data[valid_mask]
no2_re_valid = predict_matching_station [valid_mask]

# R 값 계산
correlation_matrix = np.corrcoef(obs_data_valid, no2_re_valid)
correlation_coefficient = correlation_matrix[0, 1]
print(correlation_coefficient)

df_interpolated = pd.DataFrame({
    'stnID': obs_stn,
    'latitude': obs_lat,
    'longitude': obs_lon,
    'NO2_concentration': predict_matching_station
})

print(df_interpolated)

plt.figure(figsize=(10, 10))
plt.scatter(obs_data, predict_matching_station, c='blue', edgecolor='k', alpha=0.6)
plt.xlabel('Observed NO2 Concentration (ppb)')
plt.ylabel('CMAQ Interpolated NO2 Concentration (ppb)')
plt.title('Comparison of Observed and CMAQ Interpolated NO2 Concentrations')
plt.plot([min(obs_data), max(obs_data)], [min(obs_data), max(obs_data)], 'r--', linewidth=2)
plt.grid(True)
plt.show()

#%%

obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

no2_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def,data=NO2, target_geo_def=obs_def, 
    radius_of_influence=15000, neighbours=2, weight_funcs=wf, fill_value=None)


if obs_data.shape != no2_re.shape:
    min_len = min(len(obs_data), len(no2_re))
    obs_data = obs_data[:min_len]
    no2_re = no2_re[:min_len]

# NaN 값이 있는 행 제외하고 상관 계수 계산
valid_mask = ~np.isnan(obs_data) & ~np.isnan(no2_re)
obs_data_valid = obs_data[valid_mask]
no2_re_valid = no2_re[valid_mask]

# R 값 계산
correlation_matrix = np.corrcoef(obs_data_valid, no2_re_valid)
correlation_coefficient = correlation_matrix[0, 1]
print(correlation_coefficient)

plt.figure(figsize=(10, 10))
plt.scatter(obs_data, no2_re, c='blue', edgecolor='k', alpha=0.6)
plt.xlabel('Observed NO2 Concentration (ppb)')
plt.ylabel('CMAQ Interpolated NO2 Concentration (ppb)')
plt.title('Comparison of Observed and CMAQ Interpolated NO2 Concentrations')
plt.plot([min(obs_data), max(obs_data)], [min(obs_data), max(obs_data)], 'r--', linewidth=2)
plt.grid(True)
plt.show()

df_interpolated = pd.DataFrame({
    'stnID': obs_stn,
    'latitude': obs_lat,
    'longitude': obs_lon,
    'NO2_concentration': no2_re
})

print(df_interpolated)

#%%
#%%
#stn_nums = [col.split('_')[0] for col in data_obs.columns]
#lon_AK = [float(col.split('_')[1]) for col in data_obs.columns]
#lat_AK = [float(col.split('_')[2]) for col in data_obs.columns]

data_obs_daily = data_obs.resample('D').mean()

