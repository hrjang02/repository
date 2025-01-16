#%%
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from netCDF4 import Dataset
import xarray as xr
import pyresample
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import calendar


#%%
'''
공간분포 값 
'''



years = '2023'
months = [f'{m:02d}' for m in range(1, 13)]

path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{years}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values
lon_CMAQ = path_cmaq['LON'].squeeze().values

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(55, 45), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

# 제외할 날짜 리스트
excluded_dates = [
    '2023-01-01', '2023-01-02', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
    '2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05', '2023-06-06',
    '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10', '2023-07-25', '2023-07-26',
    '2023-07-27', '2023-07-30', '2023-12-02', '2023-12-03'
]
excluded_dates = [pd.to_datetime(date) for date in excluded_dates]



for i, month in enumerate(months):
    file = pd.read_csv(f'/home/intern/{years}_O3.csv')
    obs_stn = file.columns[2:].str.split('_').str[0]
    obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
    obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

    file['측정일시'] = pd.to_datetime(file['측정일시'])
    file.set_index('측정일시', inplace=True)

    # 달마다 마지막 날짜를 계산
    _, last_day = calendar.monthrange(int(years), int(month))
    file = file.loc[f'{years}-{month}-01':f'{years}-{month}-{last_day}']
    file = file[file.index.hour == 0]
    file = file.drop(columns='Unnamed: 0')

    merge_list = []

    for date in pd.date_range(start=f'{years}-{month}-01', end=f'{years}-{month}-{last_day}', freq='D'):
        if date in excluded_dates:
            continue

        if date <= pd.to_datetime('2023-07-24'):
            path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
            time_index = 15 
        else:
            path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v532_cb6_DONGNAM_D02_{date.strftime("%Y%m%d")}.nc'
            time_index = 38  


        cmaq_o3 = Dataset(path_cmaq_o3, mode='r')


        O3 = np.squeeze(cmaq_o3.variables['O3'][time_index, 0, :, :])

        obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
        CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

        wf = lambda r: 1 / r ** 2

        O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                                radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                                fill_value=None)

        obs_data = file.loc[date]

        O3_re_flattened = O3_re.flatten()

        CMAQ_df = pd.DataFrame({
            'datetime': date,
            'station': obs_stn,
            'lon': obs_lon,
            'lat': obs_lat,
            'CMAQ_O3': O3_re_flattened,
            'OBS_O3': obs_data
        })

        merge_list.append(CMAQ_df)

    Night_o3 = pd.concat(merge_list, ignore_index=True)
    Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000
    Night_o3['Difference'] = Night_o3['CMAQ_O3'] - Night_o3['OBS_O3']

    ax = axes[i]
    scatter = ax.scatter(Night_o3['lon'], Night_o3['lat'], c=Night_o3['Difference'], cmap='coolwarm', s=100,
                          alpha=1.0, transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_title(f'{years}-{month}', fontsize = 30, fontweight = "bold")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

cbar = plt.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Ozone Concentration Difference (ppb)',fontsize = 40)

fig.suptitle(f'Spatial_Distribution_of_Midnight_Ozone_{years}  (CMAQ_O3-OBS_O3)',fontsize = 50, fontweight ="bold") 
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f'/home/intern/hrjang/figure/2023/Spatial_Distribution_of_Midnight_Ozone_{years}.png')

# %%
'''
2023년 야간 오존 validation / 오차 지수

'''

all_month_data = []

years = '2023'
months = [f'{m:02d}' for m in range(1, 13)]


# 제외할 날짜 리스트
excluded_dates = [
    '2023-01-01', '2023-01-02', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
    '2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05', '2023-06-06',
    '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10', '2023-07-25', '2023-07-26',
    '2023-07-27', '2023-07-30', '2023-12-02', '2023-12-03'
]
excluded_dates = [pd.to_datetime(date) for date in excluded_dates]


def calculate_metrics(df):
    corr_list = []
    MB_list = []
    ME_list = []
    NMB_list = []
    NME_list = []
    RMSE_list = []

    for i, row in df.iterrows():
        obs = row['OBS_O3']
        pred = row['CMAQ_O3']

        MB = np.mean(pred - obs)
        ME = np.mean(np.abs(pred - obs))
        NMB = MB / np.mean(obs)
        NME = ME / np.mean(obs)
        RMSE = np.sqrt(np.mean((pred - obs) ** 2))

        MB_list.append(MB)
        ME_list.append(ME)
        NMB_list.append(NMB)
        NME_list.append(NME)
        RMSE_list.append(RMSE)

    df['MB'] = MB_list
    df['ME'] = ME_list
    df['NMB'] = NMB_list
    df['NME'] = NME_list
    df['RMSE'] = RMSE_list

    return df


'''
CMAQ data -> 각 날짜에 대해 자정(00:00) 데이터 처리
'''

path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{years}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values


for month in months:
    file = pd.read_csv(f'/home/intern/{years}_O3.csv')
    obs_stn = file.columns[2:].str.split('_').str[0]
    obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
    obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

    file['측정일시'] = pd.to_datetime(file['측정일시'])
    file.set_index('측정일시', inplace=True)

    # 달마다 마지막 날짜를 계산
    _, last_day = calendar.monthrange(int(years), int(month))
    file = file.loc[f'{years}-{month}-01':f'{years}-{month}-{last_day}']
    file = file[file.index.hour == 0]
    file = file.drop(columns='Unnamed: 0')

    merge_list = []

    for date in pd.date_range(start=f'{years}-{month}-01', end=f'{years}-{month}-{last_day}', freq='D'):
        if date in excluded_dates:
            continue

        if date <= pd.to_datetime('2023-07-24'):
            path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
            time_index = 15 
        else:
            path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v532_cb6_DONGNAM_D02_{date.strftime("%Y%m%d")}.nc'
            time_index = 38  


        cmaq_o3 = Dataset(path_cmaq_o3, mode='r')


        O3 = np.squeeze(cmaq_o3.variables['O3'][time_index, 0, :, :])

        obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
        CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

        wf = lambda r: 1 / r ** 2

        O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                                radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                                fill_value=None)

        obs_data = file.loc[date]

        O3_re_flattened = O3_re.flatten()

        CMAQ_df = pd.DataFrame({
            'datetime': date,
            'station': obs_stn,
            'lon': obs_lon,
            'lat': obs_lat,
            'CMAQ_O3': O3_re_flattened,
            'OBS_O3': obs_data
        })

        merge_list.append(CMAQ_df)

    Night_o3 = pd.concat(merge_list, ignore_index=True)
    Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000


    result_df = calculate_metrics(Night_o3.groupby('station').mean())


    fig, axes = plt.subplots(3, 2, figsize=(15, 18), subplot_kw={'projection': ccrs.PlateCarree()})

    metrics = {
        'MB': 'Mean Bias (MB)',
        'ME': 'Mean Error (ME)',
        'NMB': 'Normalized Mean Bias (NMB)',
        'NME': 'Normalized Mean Error (NME)',
        'RMSE': 'Root Mean Square Error (RMSE)'
    }
    units = {
        'MB': 'ppb',
        'ME': 'ppb',
        'NMB': '%',
        'NME': '%',
        'RMSE': 'ppb'
    }

    for i, (metric, title) in enumerate(metrics.items()):
        ax = axes[i // 2, i % 2]
        sc = ax.scatter(result_df['lon'], result_df['lat'], c=result_df[metric], cmap='coolwarm', s=80, transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize = 15, fontweight = "bold")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.COASTLINE)
        #ax.add_feature(cfeature.LAND, edgecolor='black')

        #gridlines = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=1.0, linestyle='--')
        #gridlines.top_labels = False
        #gridlines.right_labels = False

        fig.colorbar(sc, ax=ax, label=f'{metric} ({units[metric]})')


    axes[2, 1].axis('off')

    fig.suptitle(f'Error Metrics of Midnight O3 Concentrations {years}-{month}', fontsize = 25, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f'/home/intern/hrjang/figure/2023/Error_Metrics_of_Midnight_O3_Concentrations_{years}-{month}.png')
    
# %%
'''
2023년 야간 오존 validation

'''

years = '2023'

# 제외할 날짜 리스트
excluded_dates = [
    '2023-01-01', '2023-01-02', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
    '2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05', '2023-06-06',
    '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10', '2023-07-25', '2023-07-26',
    '2023-07-27', '2023-07-30', '2023-12-02', '2023-12-03'
]
excluded_dates = [pd.to_datetime(date) for date in excluded_dates]

# CMAQ data -> 각 날짜에 대해 자정(00:00) 데이터 처리
path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{years}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values

file = pd.read_csv(f'/home/intern/{years}_O3.csv')
obs_stn = file.columns[2:].str.split('_').str[0]
obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)

# 전체 기간 데이터를 필터링하고 00시 데이터만 선택
file = file[file.index.hour == 0]
file = file.drop(columns='Unnamed: 0')
file = file[~file.index.isin(excluded_dates)]

merge_list = []

# 월별 데이터 처리
for month in range(1, 13):
    _, last_day = calendar.monthrange(int(years), int(month))
    monthly_data = file.loc[f'{years}-{month:02d}-01':f'{years}-{month:02d}-{last_day:02d}']

    for date in pd.date_range(start=f'{years}-{month:02d}-01', end=f'{years}-{month:02d}-{last_day:02d}', freq='D'):
        if date in excluded_dates:
            continue

        if date <= pd.to_datetime('2023-07-24'):
            path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
            time_index = 15 
        else:
            path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v532_cb6_DONGNAM_D02_{date.strftime("%Y%m%d")}.nc'
            time_index = 38  

        cmaq_o3 = Dataset(path_cmaq_o3, mode='r')
        O3 = np.squeeze(cmaq_o3.variables['O3'][time_index, 0, :, :])

        obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
        CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

        wf = lambda r: 1 / r ** 2

        O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                                   radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                                   fill_value=None)

        obs_data = monthly_data.loc[date]

        O3_re_flattened = O3_re.flatten()

        CMAQ_df = pd.DataFrame({
            'datetime': date,
            'station': obs_stn,
            'lon': obs_lon,
            'lat': obs_lat,
            'CMAQ_O3': O3_re_flattened,
            'OBS_O3': obs_data.values.flatten()
        })

        merge_list.append(CMAQ_df)

Night_o3 = pd.concat(merge_list, ignore_index=True)
Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000

# 상관 계수 계산
def calculator(actual, predicted):
    ME2 = np.abs(predicted - actual)
    ME = np.mean(ME2)
    NME = (np.sum(ME2) / np.sum(actual)) * 100
    MB = np.mean(predicted - actual)
    NMB = (np.sum(predicted - actual) / np.sum(actual)) * 100
    RMSE = np.sqrt(np.mean((predicted - actual)**2))
    
    corr, _ = pearsonr(actual, predicted)
    corr = format(corr, "5.2f")

    ME = format(ME, "5.2f")
    NME = format(NME, "5.2f")
    MB = format(MB, "5.2f")
    NMB = format(NMB, "5.2f")
    RMSE = format(RMSE, "5.2f")
    return corr, MB, ME, NMB, NME, RMSE

Night_o3['datetime'] = pd.to_datetime(Night_o3['datetime']) 



for month in range(1, 13):
    month_data = Night_o3[Night_o3['datetime'].dt.month == month]
    
    obs_data_orig = month_data['OBS_O3'].values
    cmaq_data_orig = month_data['CMAQ_O3'].values

    # NaN 값이 있는거 제외
    valid_mask = ~np.isnan(obs_data_orig) & ~np.isnan(cmaq_data_orig)
    obs_data = obs_data_orig[valid_mask]
    cmaq_data = cmaq_data_orig[valid_mask]

    slope, intercept = np.polyfit(obs_data, cmaq_data, 1)
    regression_line = slope * obs_data + intercept
    plt.rc("xtick", labelsize = 15)
    plt.rc("ytick", labelsize = 15)

    fig, ax = plt.subplots(figsize=(12, 10))
    hb = ax.hexbin(obs_data, cmaq_data, gridsize=20, cmap='hot_r', mincnt=1)
    ax.set_xlabel('Observed O3 Concentration (ppb)')
    ax.set_ylabel('CMAQ Interpolated O3 Concentration (ppb)')
    ax.set_title(f'Validation of Observed and CMAQ O3 Concentrations {years}-{month:02d}', fontsize=20, fontweight="bold")
    fig.colorbar(hb, ax=ax, label='Counts')

    ax.plot(obs_data, regression_line, 'r-', label=f'Regression line\n y = {slope:.2f}x + {intercept:.2f}\n')
    ax.plot([min(obs_data), max(obs_data)], [min(obs_data), max(obs_data)], 'k--', linewidth=2, label='1:1 Line')
    values = (f'\nR value = {calculator(obs_data, cmaq_data)[0]}\n'
              f'MB = {calculator(obs_data, cmaq_data)[1]}\n'
              f'ME = {calculator(obs_data, cmaq_data)[2]}\n'
              f'NME = {calculator(obs_data, cmaq_data)[3]}\n'
              f'NMB = {calculator(obs_data, cmaq_data)[4]}\n'
              f'RMSE = {calculator(obs_data, cmaq_data)[5]}')
    ax.plot([], [], ' ', label=values)

    ax.set_xlim([0, 90])  
    ax.set_ylim([0, 90])  

    ax.grid(True)
    ax.legend(loc='upper left')
    plt.savefig(f'/home/intern/hrjang/figure/2023/Validation of Observed and CMAQ O3 Concentrations {years}-{month:02d}.png')
 

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyresample
import xarray as xr
from netCDF4 import Dataset
import calendar

year = '2023'
month = '08'


file = pd.read_csv(f'/home/intern/{year}_O3.csv')
obs_stn = file.columns[2:].str.split('_').str[0]
obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)


path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{year}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values
lon_CMAQ = path_cmaq['LON'].squeeze().values


_, last_day = calendar.monthrange(int(year), int(month))
file = file.loc[f'{year}-{month}-01':f'{year}-{month}-{last_day}']
file = file[file.index.hour == 0]
file = file.drop(columns='Unnamed: 0')


excluded_dates = [
    '2023-01-01', '2023-01-02', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
    '2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05', '2023-06-06',
    '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10', '2023-07-25', '2023-07-26',
    '2023-07-27', '2023-07-30', '2023-12-02', '2023-12-03'
]
excluded_dates = [pd.to_datetime(date) for date in excluded_dates]

merge_list = []

for date in pd.date_range(start=f'{year}-{month}-01', end=f'{year}-{month}-{last_day}', freq='D'):
    if date in excluded_dates:
        continue

    if date <= pd.to_datetime('2023-07-24'):
        path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
        time_index = 15 
    else:
        path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v532_cb6_DONGNAM_D02_{date.strftime("%Y%m%d")}.nc'
        time_index = 38  


    cmaq_o3 = Dataset(path_cmaq_o3, mode='r')


    O3 = np.squeeze(cmaq_o3.variables['O3'][time_index, 0, :, :])

    obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
    CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

    wf = lambda r: 1 / r ** 2

    O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                               radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                               fill_value=None)

    obs_data = file.loc[date]

    O3_re_flattened = O3_re.flatten()

    CMAQ_df = pd.DataFrame({
        'datetime': date,
        'station': obs_stn,
        'lon': obs_lon,
        'lat': obs_lat,
        'CMAQ_O3': O3_re_flattened,
        'OBS_O3': obs_data
    })

    merge_list.append(CMAQ_df)

Night_o3 = pd.concat(merge_list, ignore_index=True)

Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000
Night_o3['Difference'] = Night_o3['CMAQ_O3'] - Night_o3['OBS_O3']

Night_o3.set_index('datetime', inplace=True)
Night_o3_mean = Night_o3.resample('D').mean(numeric_only=True)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(Night_o3_mean.index, Night_o3_mean['OBS_O3'], label='Observed O3', color='blue')
ax.plot(Night_o3_mean.index, Night_o3_mean['CMAQ_O3'], label='CMAQ Predicted O3', color='red')

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Ozone Concentration (ppb)', fontsize=14)
ax.set_title(f'Time Series of Midnight Ozone Concentration for August {year}', fontsize=16, fontweight='bold')
ax.legend()
ax.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f'/home/intern/hrjang/figure/{year}/Time_Series_of_Midnight_Ozone_Concentration_{year}_{month}.png')


# %%
730
