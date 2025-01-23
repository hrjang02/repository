#%%
import pandas as pd
from scipy.spatial import cKDTree
from dateutil import rrule
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from scipy.interpolate import griddata
from package import calculate
#%% 
start_time = "2021-02-01 00:00"  
end_time = "2021-03-01 00:00"  
var = 'ws'
mcip_path = f'/data07/ULSAN_EHC/MCIP/202102/D02/METCRO2D_D02_EHC.nc'
grid_path = f'/data07/ULSAN_EHC/MCIP/202102/D02/GRIDCRO2D_D02_EHC.nc'
#%% ASOS+AWS 데이터 
data_stn_list = []
for year in range(2021, 2023):
    df_ASOS = pd.read_csv(f'/data01/OBS/met_orig/ASOS/data/{year}_{var}.csv')
    df_ASOS['tm'] = pd.to_datetime(df_ASOS['tm'])
    df_AWS = pd.read_csv(f'/data01/OBS/met_orig/AWS/data/{year}_{var}.csv')
    df_AWS['tm'] = pd.to_datetime(df_AWS['tm'])
    df_stn_year = pd.concat([df_ASOS, df_AWS], axis=1)
    df_stn_year = df_stn_year.loc[:, ~df_stn_year.columns.duplicated()]

    data_stn_list.append(df_stn_year)
    
df_stn = pd.concat(data_stn_list)
df_stn.set_index('tm', inplace=True)
df_stn.index = df_stn.index.tz_localize('Asia/Seoul')
data_stn_KST = df_stn.loc[start_time:end_time]

obs_stn = data_stn_KST.columns[:].str.split('_').str[0].tolist() 
obs_lon = data_stn_KST.columns[:].str.split('_').str[1].astype(float)
obs_lat = data_stn_KST.columns[:].str.split('_').str[2].astype(float)
# %%
# 수정된 MCIP data + 위경도 정보 -------------------------------------------------------------
mcip_data = Dataset(mcip_path , mode='r')
daily_TFLAG_MCIP = mcip_data.variables['TFLAG'][:, 0, :]

df = []
for i in range(len(daily_TFLAG_MCIP)):
    mcip_ws = mcip_data.variables['WSPD10'][i, 0, :, :]
    grid_data = xr.open_dataset(grid_path)
    lat_MCIP = grid_data['LAT'].squeeze().values  
    lon_MCIP = grid_data['LON'].squeeze().values

    lon_flat = lon_MCIP.ravel()
    lat_flat = lat_MCIP.ravel()
    mcip_flat = mcip_ws.ravel()

    predict_matching_station = griddata((lon_flat, lat_flat), mcip_flat, (obs_lon, obs_lat), method='cubic')

    datetime_mcip = pd.to_datetime(
        daily_TFLAG_MCIP[i, 0] * 1000000 + daily_TFLAG_MCIP[i, 1],
        format='%Y%j%H%M%S'
    )
    matching_df = pd.DataFrame([predict_matching_station], index=[datetime_mcip], columns=obs_stn)
    df.append(matching_df)
mcip_df = pd.concat(df)
#%%
mcip_df.index = mcip_df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
mcip_KST = mcip_df.loc[start_time:end_time]
#%%
data_stn_mean = data_stn_KST.mean(axis=1).values
mcip_mean = mcip_KST.mean(axis=1).values
#%% 통계지표
rmse, r, r2, mb, me, nmb, nme, model_mean, obs_mean = calculate(data_stn_mean, mcip_mean)
print(f"num: {len(mcip_mean)}\nRMSE: {rmse:.4f}, corr: {r:.4f}, R2: {r2:.4f}\nMB: {mb:.4f}, ME: {me:.4f}, NMB: {nmb:.4f}, NME: {nme:.4f}\nModel Mean: {model_mean:.4f}, Obs Mean: {obs_mean:.4f}")

# %%
