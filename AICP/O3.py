#%%
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import pyresample
import xarray as xr

# Observation의 station,위경도 불러오기
df_info = pd.read_csv('/home/hrjang/practice/data/2021_NO2.csv')
df_info['측정일시'] = pd.to_datetime(df_info['측정일시'])

obs_stn = df_info.columns[2:].str.split('_').str[0]
obs_lon = df_info.columns[2:].str.split('_').str[1].astype(float)
obs_lat = df_info.columns[2:].str.split('_').str[2].astype(float)

# Observation의 data 불러오기
year = '2021'
var = 'O3'
csv_directory = '/home/hrjang/practice/data'
file_path = f'{csv_directory}/{year}_{var}.csv'

# 시간대 변환하고 오름차순으로 정렬하기
df_obs = pd.read_csv(file_path)
df_obs['Datetime'] = pd.to_datetime(df_obs['Datetime'])
df_obs.set_index('Datetime', inplace=True)
df_obs.index = df_obs.index.tz_localize('Asia/Seoul').tz_convert('UTC')
df = df_obs.sort_values(by=['Datetime', 'station'])

# CMAQ 위경도 정보
path_cmaq = xr.open_dataset(f'/home/hrjang/practice/data/GRIDCRO2D_D02_{year}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values

merge_list = []

#4월 1일 ~ 6월 30일까지의 데이터 붙여주기
for date in pd.date_range(start=f'{year}-04-01', end=f'{year}-06-30', freq='D'):
    for hour in range(0, 24):
        path_cmaq_no2 = f'/data04/smkim/2123_AMJ_CMAQ_ERA5/CCTM_ACONC_v54_intel_D02_{year}_{date.strftime("%Y%m%d")}.nc'
        data_no2 = Dataset(path_cmaq_no2, mode='r')
        O3 = np.squeeze(data_no2.variables['O3'][hour, 0, :, :])
        
        obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
        CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

        wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

        O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                                   radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                                   fill_value=None)
        
        CMAQ_df = pd.DataFrame({'station': obs_stn, 'CMAQ_O3': O3_re})
        CMAQ_df['station'] = CMAQ_df['station'].astype(int)

        target_date = pd.to_datetime(f'{date.strftime("%Y-%m-%d")} {hour:02d}:00:00+00:00')
        filtered_df = df[df.index == target_date]

        merge_inner = pd.merge(filtered_df, CMAQ_df, how='inner', on='station')
        merge_inner['Datetime'] = target_date.strftime('%Y-%m-%d %H:00:00+00:00')

        cols = ['Datetime'] + [col for col in merge_inner if col != 'Datetime']
        merge_inner = merge_inner[cols]

        merge_list.append(merge_inner)

# 모든 데이터프레임 병합
merged_df = pd.concat(merge_list, ignore_index=True)

#merged_df = merged_df.drop(columns=['Unnamed: 3', 'Unnamed: 4'])
print("Merged DataFrame:")
print(merged_df)



#%% 일단 무시,,,
df_info = pd.read_csv('/home/hrjang/practice/data/2021_NO2.csv')
df_info['측정일시'] = pd.to_datetime(df_info['측정일시'])

obs_stn = df_info.columns[2:].str.split('_').str[0]
obs_lon = df_info.columns[2:].str.split('_').str[1].astype(float)
obs_lat = df_info.columns[2:].str.split('_').str[2].astype(float)

# Observation data + 위경도 정보
year = '2021'
var = 'O3'
csv_directory = '/home/hrjang/practice/data'
file_path = f'{csv_directory}/{year}_{var}.csv'


df_obs = pd.read_csv(file_path)
df_obs['Datetime'] = pd.to_datetime(df_obs['Datetime'])
df_obs.set_index('Datetime', inplace=True)
df_obs.index = df_obs.index.tz_localize('Asia/Seoul').tz_convert('UTC')

df = df_obs.sort_values(by=['Datetime','station']) 
print(df)

# CMAQ data + 위경도 정보 -------------------------------------------------------------
path_cmaq = xr.open_dataset(f'/home/hrjang/practice/data/GRIDCRO2D_D02_{year}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values

path_cmaq_no2 = '/data04/smkim/2123_AMJ_CMAQ_ERA5/CCTM_ACONC_v54_intel_D02_2021_20210401.nc'
data_no2 = Dataset(path_cmaq_no2, mode='r')

merge_list=[]
for date in pd.date_range(start='2021-04-01', end='2021-06-30', freq='D'):
    # 각 시간별로 반복 (0시부터 23시까지)
    for hour in range(0, 24):
        O3 = np.squeeze(data_no2.variables['O3'][hour, 0, :, :])

        obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
        CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

        wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

        O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def, 
                                                   radius_of_influence=15000, neighbours=2, weight_funcs=wf, 
                                                   fill_value=None)
        CMAQ_df = pd.DataFrame({'station': obs_stn, 'CMAQ_O3': O3_re})

        CMAQ_df['station'] = CMAQ_df['station'].astype(int)
        target_date = pd.to_datetime(f'{date.strftime("%Y-%m-%d")} {hour:02d}:00:00+00:00')
        filtered_df = df[df.index == target_date]
        merge_inner = pd.merge(filtered_df, CMAQ_df, how='inner', on='station')
        merge_inner['Datetime'] = target_date.strftime('%Y-%m-%d %H:00:00+00:00')

        cols = ['Datetime'] + [col for col in merge_inner if col != 'Datetime']
        merge_inner = merge_inner[cols]

        merge_list.append(merge_inner)

# 모든 데이터프레임을 하나로 합치기
merged_df = pd.concat(merge_list, ignore_index=True)

print("Merged DataFrame:")
print(merged_df)


# %% 파일 저장
csv_filename = f'/home/hrjang/practice/{year}_O3.csv'  
merged_df.to_csv(csv_filename, index=False)

print(f"Saved merged data to {csv_filename}")
# %%
import matplotlib.pyplot as plt

obs_data = merged_df.iloc[:, 2].values
cmaq_data = merged_df.iloc[:, 3].values

# NaN 값이 있는거 제외
valid_mask = ~np.isnan(obs_data) & ~np.isnan(cmaq_data)
obs_data_valid = obs_data[valid_mask]
cmaq_data_valid = cmaq_data[valid_mask]

# 상관 계수 계산
correlation_matrix = np.corrcoef(obs_data_valid, cmaq_data_valid)
correlation_coefficient = correlation_matrix[0, 1]
print(f'상관 계수 (R): {correlation_coefficient}')


slope, intercept = np.polyfit(obs_data_valid, cmaq_data_valid, 1)
regression_line = slope * obs_data_valid + intercept


fig, ax = plt.subplots(figsize=(10, 10))
hb = ax.hexbin(obs_data_valid, cmaq_data_valid, gridsize=50, cmap='Blues', mincnt=1)
ax.set_xlabel('Observed NO2 Concentration (ppb)')
ax.set_ylabel('CMAQ Interpolated NO2 Concentration (ppb)')
ax.set_title('Comparison of Observed and CMAQ Interpolated NO2 Concentrations')
fig.colorbar(hb, ax=ax, label='Counts')


ax.plot(obs_data_valid, regression_line, 'r-', label=f'Regression line\ny = {slope:.2f}x + {intercept:.2f}')
ax.plot([min(obs_data_valid), max(obs_data_valid)], [min(obs_data_valid), max(obs_data_valid)], 'g--', linewidth=2, label='1:1 Line')

ax.set_xlim([0, 0.10])  
ax.set_ylim([0, 0.10])  

ax.grid(True)
ax.legend()
plt.show()
# %%
