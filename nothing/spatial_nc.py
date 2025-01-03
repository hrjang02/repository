#%%
## 승미 언니가 준 파일
import numpy as np
import pandas as pd
import xarray as xr
import pyresample

# 필요한 데이터 및 정의 가져오기
# CMAQ 데이터셋 정의 및 불러오기
CMAQ_def = pyresample.geometry.GridDefinition(lons=np.linspace(lon_min, lon_max, num_lon), lats=np.linspace(lat_min, lat_max, num_lat))

# 관측소 위치 데이터 및 정의
# stn_nums와 관련된 관측소 위치 데이터 설정

data_reCMAQ = pd.DataFrame(columns=stn_nums)

for date in date_str_list:
    ACONC_CMAQ = xr.open_dataset(f'/home/hrjang/main/data/cctm/CCTM_ACONC_v54_intel_D02_2022_{date}.nc')
    for i in range(24):
        datetime = f'{date}{i:02}'
        print(datetime)
        ozone_CMAQ = np.squeeze((ACONC_CMAQ.variables['O3'][i,0,:,:])).to_numpy()
        wf = lambda r: 1/r**2 #IDW method
        ozone_re = pyresample.kd_tree.resample_custom(CMAQ_def, ozone_CMAQ, AK_def, radius_of_influence=15000, neighbours=2,weight_funcs=wf, fill_value=np.nan)
        ozone_re = pd.DataFrame(ozone_re.reshape(1, -1),columns=stn_nums)
        data_reCMAQ = pd.concat([data_reCMAQ,ozone_re],axis=0,ignore_index=True)
#%%
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 필요한 데이터 가져오기

path1 = '/home/aicp/aicp419/data/CMAQ/MCIP/GRIDCRO2D_D02_2021.nc'
path2 = '/home/aicp/aicp419/data/CMAQ/CCTM/CCTM_ACONC_v54_intel_D02_2021_20210401.nc'

data1 = Dataset(path1, mode ='r')
data2 = Dataset(path2, mode = 'r')

# 위경도 정보 가져오기
lat = np.squeeze(data1.variables['LAT'][:,:])
lon = np.squeeze(data1.variables['LON'][:,:])

# 날짜 가져오기
date_info = path2.split('_')
date1 = date_info[6]
date = date1[0:8]

## 위경도 제일 큰거 제일 작은거 가져와야함,, 


fig, axes = plt.subplots(4, 6, figsize=(20, 15), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for i, ax in enumerate(axes):
    hour = i
    ozone_data = np.squeeze((data2.variables['O3'][i,0,:,:]))
    
    ax.add_feature(cfeature.COASTLINE) 
    ax.set_extent([125, 130, 33, 39]) ##이거 추가하기
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.1, 'navy'),\
        (0.2, 'blue'), (0.3, 'cornflowerblue'), (0.4, 'turquoise'), (0.5, 'yellowgreen'), (0.6, 'yellow'),\
        (0.7, 'orange'), (0.8, 'orangered'), (0.9, 'red'), (1, 'darkred')], N=80)
    
    ax.set_title('O3 concentration of %s %s:00' %(date, hour))
    smap = ax.pcolormesh(lon, lat, ozone_data, cmap=cmap, shading = 'nearest', transform=ccrs.PlateCarree())    #산점도의 색깔 범위
    smaplegend = plt.colorbar(smap, ax=ax, location='bottom', label='[ppm]')    

plt.tight_layout()

plt.savefig('/home/aicp/aicp419/hrjang/practice/O3_concentration_subplot_%s.png' % date)
plt.show()
#%%