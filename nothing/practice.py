#%%
from netCDF4 import Dataset
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader

domain = 'D02'
year = '2021'

model_path = '/home/intern/data/CMAQ'
cctm_path = f'{model_path}/CCTM'
mcip_path = f'{model_path}/MCIP'

cctm_flist = glob(f'{cctm_path}/*_{year}_*.nc')
grdcro_fname = glob(f'{mcip_path}/GRIDCRO2D_{domain}_{year}.nc')[0]
grddot_fname = f'{mcip_path}/GRIDDOT2D_D02_2022.nc'
# %% Read CCTM file
with Dataset(cctm_flist[0], 'r') as cctm:
    no2 = cctm.variables['NO2'][:,0,:,:]
    tflag = cctm.variables['TFLAG'][:,0,:]
#%% Read MCIP file
with Dataset(grdcro_fname, mode='r') as grdcro:
    lat = np.squeeze(grdcro.variables['LAT'][:])
    lon = np.squeeze(grdcro.variables['LON'][:])
with Dataset(grddot_fname, mode='r') as grddot:
    latd = np.squeeze(grddot.variables['LATD'][:])
    lond = np.squeeze(grddot.variables['LOND'][:])
# %%
from datetime import datetime
import pandas as pd
tflag_str = tflag.astype(str)
tflag_combine = np.char.add(tflag_str[:,0], np.char.zfill(tflag_str[:,1],6))
tflag_dt = pd.to_datetime(tflag_combine, format='%Y%j%H%M%S')
target_dt = '2021-04-23 05:00:00'
target_ind = np.where(tflag_dt==target_dt)[0][0]

# %%
shp_fname = "/data02/Map_shp/TL_SCCO_CTPRVN_WGS_utf8.shp"

reader = Reader(shp_fname)
geometry = reader.geometries()
kor_shape = ShapelyFeature(
    geometry, 
    ccrs.PlateCarree(), 
    edgecolor="k", 
    linewidth=1.5
)
###### Set subplot
fig, ax = plt.subplots(
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        facecolor=(1, 1, 1),
        figsize=(5,5)
    )
###### Set extent
ax.set_extent([123, 132, 33, 39])

# gl = ax.gridlines(
            # draw_labels=True, x_inline=False, y_inline=False, zorder=102)
###### add land and border line
ax.add_feature(kor_shape, linewidth=1, facecolor='none')

im =plt.pcolormesh(lond, latd, no2[target_ind,:,:], cmap='jet',
                   shading='auto'
                   )
fig.colorbar(im, ax=ax, orientation='horizontal')
im.set_clim(0,0.05)
ax.set_title(f'CMAQ {target_dt}', fontsize=25)



# %%
