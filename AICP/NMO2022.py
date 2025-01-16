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
import scipy
import geopandas as gpd

#%%
'''
관측data로만 자정 월별 그래프
'''
years = ['2021', '2022', '2023']

fig, ax = plt.subplots(figsize=(10, 5))
for year in years:

    file = pd.read_csv(f'/home/intern/{year}_O3.csv')
    

    file['측정일시'] = pd.to_datetime(file['측정일시'])
    file.set_index('측정일시', inplace=True)
    
    file = file[file.index.hour == 0]

    file = file.drop(columns='Unnamed: 0')

    file_monthly = file.resample('M').mean()
    row_mean = file_monthly.mean(axis=1)
    monthly_avg = row_mean.groupby(row_mean.index.month).mean()
    ax.plot(monthly_avg.index, monthly_avg, label=f'{year}')


ax.set_xticks(range(1, 13))  # x축을 1~12로 설정
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # x축 레이블 설정
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Monthly O3 Concentration',fontsize=15)
ax.set_title('Monthly Average Midnight Ozone Concentration',fontsize=25,fontweight="bold")
ax.legend(loc='upper right', fontsize=15)
ax.grid(True)

plt.savefig(f'/home/intern/hrjang/figure/Monthly_Average_Midnight_Ozone_Concentration.png')

#%%
'''
공간분포 값(CMAQ-OBS)
(/home/intern/hrjang/figure/2022/Spatial_Distribution_of_Midnight_Ozone_2022.png)
'''

years = '2022'
months = [f'{m:02d}' for m in range(4, 12)]  # 4월부터 11월까지

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(30, 15), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{years}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values

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
        if date.date() == pd.to_datetime(f'{years}-08-17').date():
            continue
        path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
        cmaq_o3 = Dataset(path_cmaq_o3, mode='r')
        O3 = np.squeeze(cmaq_o3.variables['O3'][14, 0, :, :])  # 00시 데이터만 선택 (전날 15시 date 가져오면 됨)

        obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
        CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

        wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

        O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                                   radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                                   fill_value=None)

        obs_date_str = date
        obs_data = file.loc[obs_date_str]
        O3_re_flattened = O3_re.flatten()

        CMAQ_df = pd.DataFrame({'datetime':obs_date_str, 'station': obs_stn, 'lon': obs_lon, 'lat': obs_lat, 'CMAQ_O3': O3_re, 'OBS_O3': obs_data})
        merge_list.append(CMAQ_df)

    Night_o3 = pd.concat(merge_list, ignore_index=True)
    Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000
    Night_o3['Difference'] = Night_o3['CMAQ_O3'] - Night_o3['OBS_O3']

    ax = axes[i]
    scatter = ax.scatter(Night_o3['lon'], Night_o3['lat'], c=Night_o3['Difference'], cmap='coolwarm',
                         alpha=1.0, transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_title(f'{years}-{month}',fontsize = 20, fontweight = "bold")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

cbar = plt.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Ozone Concentration Difference (ppb)',fontsize = 30)

fig.suptitle(f'Spatial_Distribution_of_Midnight_Ozone_{years}_04-11 (CMAQ_O3-OBS_O3)', fontsize = 40, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.savefig(f'/home/intern/hrjang/figure/2022/Spatial_Distribution_of_Midnight_Ozone_{years}.png')


#%%
'''
2022년 야간 오존 validation / 오차 지수
4월에서 11월까지만(/home/intern/hrjang/figure/2022/Error_Metrics_of_Midnight_O3_Concentrations 2022-09.png)
'''

year = '2022'
month = '11'
##ㄴOBS/CMAQ 날짜 변경해주기

'''
Observation의 data 불러오기(시간 : KST) -> 자정(00:00)의 시간만 가져오기
'''

file= pd.read_csv(f'/home/intern/{year}_O3.csv')
obs_stn = file.columns[2:].str.split('_').str[0]
obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)


file = file.loc[f'{year}-{month}-01':f'{year}-{month}-30']
file = file[file.index.hour == 0]
file = file.drop(columns='Unnamed: 0')

file = file[file.index.date != pd.to_datetime(f'{year}-08-17').date()]


'''
CMAQ data -> 각 날짜에 대해 자정(00:00) 데이터 처리
'''

path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{year}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values


merge_list = []
#각 날짜별 KST의 0시의 데이터를 가져옴
for date in pd.date_range(start=f'{year}{month}01', end=f'{year}{month}30', freq='D'):
    # 8월 17일 날짜
    if date.date() == pd.to_datetime(f'{year}-08-17').date():
        continue
    
    path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
    cmaq_o3 = Dataset(path_cmaq_o3, mode='r')
    O3 = np.squeeze(cmaq_o3.variables['O3'][14, 0, :, :])  # 00시 데이터만 선택 (전날 15시 date 가져오면 됨)
    
    obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
    CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

    wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

    O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                               radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                               fill_value=None)
    
    obs_date_str = date
    obs_data = file.loc[obs_date_str]
    O3_re_flattened = O3_re.flatten()
    
    CMAQ_df = pd.DataFrame({'datetime':obs_date_str, 'station': obs_stn, 'lon': obs_lon, 'lat': obs_lat, 'CMAQ_O3': O3_re, 'OBS_O3': obs_data})
    merge_list.append(CMAQ_df)

Night_o3 = pd.concat(merge_list, ignore_index=True)

Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000


'''
상관관계 지표
'''

Night_o3 = Night_o3.groupby('station').mean()
##상관계수

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


result_df = calculate_metrics(Night_o3)



#상관관계 그림 그리기
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
    
    gridlines = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gridlines.top_labels = False
    gridlines.right_labels = False
    
    fig.colorbar(sc, ax=ax, label=f'{metric} ({units[metric]})')


axes[2, 1].axis('off')

fig.suptitle(f'Error Metrics of Midnight O3 Concentrations {years}-{month}', fontsize = 25, fontweight="bold")
plt.tight_layout() 
plt.savefig(f'/home/intern/hrjang/figure/2022/Error_Metrics_of_Midnight_O3_Concentrations {year}-{month}.png')

    
#%%
'''
2022년 야간 오존 validation
4월에서 11월까지만(/home/intern/hrjang/figure/2022/Validation of Observed and CMAQ O3 Concentrations 2022-06.png)
'''
year = '2022'
months = ['04', '05', '06', '07', '08', '09', '10', '11']

# 제외할 날짜 리스트
excluded_dates = ['2022-08-17']
excluded_dates = [pd.to_datetime(date) for date in excluded_dates]

# 관측 데이터 불러오기
file = pd.read_csv(f'/home/intern/{year}_O3.csv')
obs_stn = file.columns[2:].str.split('_').str[0]
obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)
file = file[file.index.date != pd.to_datetime(f'{year}-08-17').date()]

# CMAQ 데이터 불러오기
path_cmaq = xr.open_dataset(f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{year}.nc')
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values

merge_list = []

for month in months:
    _, last_day = calendar.monthrange(int(year), int(month))
    file_month = file.loc[f'{year}-{month}-01':f'{year}-{month}-{last_day}']
    file_month = file_month[file_month.index.hour == 0]
    file_month = file_month.drop(columns='Unnamed: 0')
    file_month = file_month[~file_month.index.isin(excluded_dates)]
    
    for date in pd.date_range(start=f'{year}-{month}-01', end=f'{year}-{month}-{last_day}', freq='D'):
        if date in excluded_dates:
            continue
        
        path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
        cmaq_o3 = Dataset(path_cmaq_o3, mode='r')
        O3 = np.squeeze(cmaq_o3.variables['O3'][14, 0, :, :])  # 00시 데이터만 선택 (전날 15시 date 가져오면 됨)
        
        obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
        CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

        wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

        O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                                   radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                                   fill_value=None)
        
        obs_date_str = date
        obs_data = file_month.loc[obs_date_str]
        O3_re_flattened = O3_re.flatten()
        
        CMAQ_df = pd.DataFrame({'datetime': obs_date_str, 'station': obs_stn, 'lon': obs_lon, 'lat': obs_lat, 'CMAQ_O3': O3_re, 'OBS_O3': obs_data})
        merge_list.append(CMAQ_df)

Night_o3 = pd.concat(merge_list, ignore_index=True)
Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000

obs_data_orig = Night_o3.iloc[:, 5].values
cmaq_data_orig = Night_o3.iloc[:, 4].values

# NaN 값이 있는거 제외
valid_mask = ~np.isnan(obs_data_orig) & ~np.isnan(cmaq_data_orig)
obs_data = obs_data_orig[valid_mask]
cmaq_data = cmaq_data_orig[valid_mask]

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


slope, intercept = np.polyfit(obs_data, cmaq_data, 1)
regression_line = slope * obs_data + intercept

fig, ax = plt.subplots(figsize=(12, 10))
hb = ax.hexbin(obs_data, cmaq_data, gridsize=20, cmap='hot_r', mincnt=1)
ax.set_xlabel('Observed O3 Concentration (ppb)')
ax.set_ylabel('CMAQ Interpolated O3 Concentration (ppb)')
ax.set_title(f'Validation of Observed and CMAQ O3 Concentrations {year} (April to November)', fontsize = 12, fontweight = "bold")
fig.colorbar(hb, ax=ax, label='Counts')

ax.plot(obs_data, regression_line, 'r-', label=f'Regression line\ny = {slope:.2f}x + {intercept:.2f}\n')
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
plt.savefig(f'/home/intern/hrjang/figure/2022/Validation of Observed and CMAQ O3 Concentrations {year} (April to November).png')


# %%

year = '2022'
month = '11'
obs_file = f'/home/intern/{year}_O3.csv'
grid_file = f'/home/intern/data/CMAQ/MCIP/GRIDCRO2D_D02_{year}.nc'


file = pd.read_csv(obs_file)
obs_stn = file.columns[2:].str.split('_').str[0]
obs_lon = file.columns[2:].str.split('_').str[1].astype(float)
obs_lat = file.columns[2:].str.split('_').str[2].astype(float)

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)

file = file.loc[f'{year}-{month}-01':f'{year}-{month}-30']
file = file[file.index.hour == 0]
file = file.drop(columns='Unnamed: 0')

file = file[file.index.date != pd.to_datetime(f'{year}-08-17').date()]


path_cmaq = xr.open_dataset(grid_file)
lat_CMAQ = path_cmaq['LAT'].squeeze().values  
lon_CMAQ = path_cmaq['LON'].squeeze().values

merge_list = []
for date in pd.date_range(start=f'{year}{month}01', end=f'{year}{month}30', freq='D'):
    if date.date() == pd.to_datetime(f'{year}-08-17').date():
        continue
    
    path_cmaq_o3 = f'/data04/SIJAQ_OUT/CMAQ/Forecast_combine/COMBINE_CONC_v52_cb6r3_ae6_D02_{date.strftime("%Y%m%d")}.nc'
    cmaq_o3 = Dataset(path_cmaq_o3, mode='r')
    O3 = np.squeeze(cmaq_o3.variables['O3'][14, 0, :, :])  # 00시 데이터만 선택 (전날 15시 date 가져오면 됨)
    
    obs_def = pyresample.geometry.SwathDefinition(lons=obs_lon.values, lats=obs_lat.values)
    CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

    wf = lambda r: 1 / r ** 2  # 역거리 가중치 함수

    O3_re = pyresample.kd_tree.resample_custom(source_geo_def=CMAQ_def, data=O3, target_geo_def=obs_def,
                                               radius_of_influence=15000, neighbours=2, weight_funcs=wf,
                                               fill_value=None)
    
    obs_date_str = date
    obs_data = file.loc[obs_date_str]
    O3_re_flattened = O3_re.flatten()
    
    CMAQ_df = pd.DataFrame({'datetime': obs_date_str, 'station': obs_stn, 'lon': obs_lon, 'lat': obs_lat, 'CMAQ_O3': O3_re, 'OBS_O3': obs_data})
    merge_list.append(CMAQ_df)

Night_o3 = pd.concat(merge_list, ignore_index=True)
Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000

Night_o3 = Night_o3.groupby('station').mean()

#%%

def calculator(actual, predicted):
    ME2 = np.abs(predicted - actual)
    ME = np.mean(ME2)
    NME = (np.sum(ME2) / np.sum(actual)) * 100
    MB = np.mean(predicted - actual)
    NMB = (np.sum(predicted - actual) / np.sum(actual)) * 100
    RMSE = np.sqrt(np.mean((predicted - actual) ** 2))
    corr = scipy.stats.spearmanr(actual, predicted)  # 피어슨 상관계수 계산
    
    # 포맷팅
    corr = format(corr, "5.2f")
    ME = format(ME, "5.2f")
    NME = format(NME, "5.2f")
    MB = format(MB, "5.2f")
    NMB = format(NMB, "5.2f")
    RMSE = format(RMSE, "5.2f")

    return corr, RMSE, MB, NMB, ME, NME

# 결과를 저장할 딕셔너리 초기화
results = {
    'lon': [],
    'lat': [],
    'R': [],
    'RMSE': [],
    'MB': [],
    'NMB': [],
    'ME': [],
    'NME': []
}

# 각 행에 대해 지표 계산
for _, row in Night_o3.iterrows():
    obs_data = np.array([row['OBS_O3']])
    cmaq_data = np.array([row['CMAQ_O3']])
    
    # 상관계수 계산을 위한 조건 체크
    if len(obs_data) < 2 or len(cmaq_data) < 2:
        # 단일 값에 대해서는 계산 생략
        continue
    
    stats = calculator(obs_data, cmaq_data)
    
    results['lon'].append(row['lon'])
    results['lat'].append(row['lat'])
    results['R'].append(float(stats[0]))
    results['RMSE'].append(float(stats[1]))
    results['MB'].append(float(stats[2]))
    results['NMB'].append(float(stats[3]))
    results['ME'].append(float(stats[4]))
    results['NME'].append(float(stats[5]))

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 공간 분포 지도 시각화 함수
def plot_spatial_distribution(gdf, column, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    base = world.plot(ax=ax, color='white', edgecolor='black')
    gdf.plot(column=column, ax=base, legend=True, legend_kwds={'label': title, 'orientation': 'horizontal'})
    plt.title(title)
    plt.show()

# GeoDataFrame으로 변환
gdf = gpd.GeoDataFrame(results_df, geometry=gpd.points_from_xy(results_df.lon, results_df.lat))

# 각 통계 지표에 대한 공간 분포 지도 생성
plot_spatial_distribution(gdf, 'R', 'Spatial Distribution of Correlation (R)')
plot_spatial_distribution(gdf, 'RMSE', 'Spatial Distribution of RMSE')
plot_spatial_distribution(gdf, 'MB', 'Spatial Distribution of MB')
plot_spatial_distribution(gdf, 'NMB', 'Spatial Distribution of NMB')
plot_spatial_distribution(gdf, 'ME', 'Spatial Distribution of ME')
plot_spatial_distribution(gdf, 'NME', 'Spatial Distribution of NME')
# %%
