#%%
import geopandas as gpd
import pandas as pd
import shapely
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
#%%
GEMS_O3_path = '/data02/SATELLITE/GEMS/v2.1.0/O3T/202502/GK2_GEMS_L2_20250210_0645_O3T_FW_DPRO_ORI.nc'
# GEMS_NO2_path = '/data02/SATELLITE/GEMS/v3.0.0/NO2/202502/GK2_GEMS_L2_20250210_0645_NO2_FW_DPRO_ORI.nc'

shp_path = '/data02/Map_shp/TL_SCCO_CTPRVN_WGS_utf8.shp'
product ='O3T'
# %%
GEMS_O3T = Dataset(GEMS_O3_path, mode='r')

gems_lats = GEMS_O3T.groups['Geolocation Fields'].variables['Latitude'][:]
gems_lons = GEMS_O3T.groups['Geolocation Fields'].variables['Longitude'][:]

Flags = GEMS_O3T.groups['Data Fields'].variables['FinalAlgorithmFlags'][:]
Cfraction = GEMS_O3T.groups['Data Fields'].variables['EffectiveCloudFraction'][:]
TC_O3 = GEMS_O3T.groups['Data Fields'].variables['ColumnAmountO3'][:]

plot_text = 'O$_3$'
unit = 'DU'

## quality control ##
TC_O3[TC_O3 < 50] = np.nan
TC_O3[TC_O3 > 700] = np.nan
TC_O3[Cfraction > 0.33] = np.nan
TC_O3[Flags != 0] = np.nan
#%%
from air_toolbox import gems, plot, util, geo

GEMS_O3_path = '/data02/SATELLITE/GEMS/v2.1.0/O3T/202502/GK2_GEMS_L2_20250210_0645_O3T_FW_DPRO_ORI.nc'
gems_f = gems.read_gems(GEMS_O3_path)
#%% pcolormesh할 때는 lon_plot, lat_plot 사용
fig, ax = plot.map_plot(preset='KRCN')
ax.pcolormesh(gems_f.lon_plot, gems_f.lat_plot, gems_f.tcd, cmap='jet')
#%%
fig, ax = plot.map_plot(preset='Ulsan', extent= plot.domain_info('Onsan'))
ax.scatter(gems_f.lon, gems_f.lat, c='k') ## grid center 값
ax.scatter(gems_f.lon_plot, gems_f.lat_plot, c='r') ## Interpolated된 corner값

result, ind_0_list, ind_1_list = geo.domain_test(plot.domain_info('Onsan'), gems_f.lon, gems_f.lat, return_index=True)
for i in ind_0_list:
    for j in ind_1_list:
        ax.text(gems_f.lon[i,j], gems_f.lat[i,j], str(i)+','+str(j), fontsize=15) 

        ax.text(gems_f.lon_corner[i,j,0], gems_f.lat_corner[i,j,0], f'{i}, {j}, 0', fontsize=15)
        ax.text(gems_f.lon_corner[i,j,1], gems_f.lat_corner[i,j,1], f'{i}, {j}, 1', fontsize=15)
        ax.text(gems_f.lon_corner[i,j,2], gems_f.lat_corner[i,j,2], f'{i}, {j}, 2', fontsize=15)
        ax.text(gems_f.lon_corner[i,j,3], gems_f.lat_corner[i,j,3], f'{i}, {j}, 3', fontsize=15)

#%%
