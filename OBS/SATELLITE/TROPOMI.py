#%%
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
#%%
file_path = '/data02/SATELLITE/TROPOMI/Products/OFFL/O3/2022/07/'
file_name = 'S5P_OFFL_L2__O3_____*'

files = glob(file_path + file_name)
#%%
file = Dataset(file_path + file_name, 'r')
product_group = file.groups['PRODUCT']
#%%

o3 = product_group.variables['ozone_total_vertical_column'][0, :, :]
lat = product_group.variables['latitude'][0, :, :]
lon = product_group.variables['longitude'][0, :, :]
qa = product_group.variables['qa_value'][0, :, :]

mask = (qa >= 0.5)
o3_masked = np.where(mask, o3, np.nan)

plt.figure(figsize=(12, 6))
plt.imshow(o3_masked, origin='lower', cmap='viridis',
           extent=[np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)],
           aspect='auto')
plt.colorbar(label='Ozone Column (mol/m²)')
plt.title('TROPOMI Ozone Total Vertical Column')
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# 원본 데이터
o3 = product_group.variables['ozone_total_vertical_column'][0, :, :]
lat = product_group.variables['latitude'][0, :, :]
lon = product_group.variables['longitude'][0, :, :]
qa = product_group.variables['qa_value'][0, :, :]

# 마스킹
mask = (qa >= 0.5)
o3_masked = np.where(mask, o3, np.nan)

# 대한민국 범위 인덱스 마스크 생성
korea_mask = (
    (lat >= 33) & (lat <= 39) &
    (lon >= 125) & (lon <= 130)
)

# 해당 영역만 자르기
o3_korea = np.where(korea_mask, o3_masked, np.nan)
lat_korea = np.where(korea_mask, lat, np.nan)
lon_korea = np.where(korea_mask, lon, np.nan)

# extent 계산 (NaN 제거하고 최솟값 ~ 최댓값)
lat_min, lat_max = np.nanmin(lat_korea), np.nanmax(lat_korea)
lon_min, lon_max = np.nanmin(lon_korea), np.nanmax(lon_korea)

# 시각화
plt.figure(figsize=(10, 6))
plt.imshow(o3_korea, origin='lower', cmap='viridis',
           extent=[lon_min, lon_max, lat_min, lat_max],
           aspect='auto')
plt.colorbar(label='Ozone Column (mol/m²)')
plt.title('TROPOMI Ozone Column over South Korea')
plt.tight_layout()
plt.show()

# %%
