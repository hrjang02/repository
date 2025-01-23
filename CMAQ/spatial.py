from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

#%%
path = '/home/aicp/aicp419/data/CMAQ'
GRID_data = path+'/MCIP/GRIDCRO2D_D02_2021.nc'
CMAQ_geo = Dataset(GRID_data, mode = 'r') 

lat_np = np.squeeze(CMAQ_geo.variables['LAT'][:])
lon_np = np.squeeze(CMAQ_geo.variables['LON'][:])
lon_min, lon_max, lat_min, lat_max = lon_np.min(), lon_np.max(), lat_np.min(), lat_np.max()

#%%
Dates = ['01','04','07','10']
VAR_asl = ['O3','NO2','NH3']
VAR_asl_T = ['O$_3$','NO$_2$','NH$_3$']  # 'PM$_2$$_.$$_5$' _<= 아래첨자
UNIT_asl = [12,20,40,20,25]
custom_cmap = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#D0FCFC'),
    (0.2, '#32B8FE'),
    (0.4, '#95EB3F'),
    (0.6, '#FFF980'),
    (0.8, '#FFA200'),
    (1, '#FC1C01'),
], N=256)
extent = [125, 130, 33, 39]
grid_lon_dist = 1 
grid_lat_dist = 1
for D in Dates:
    CMAQ_FileName = f'CCTM_ACONC_v54_intel_D02_2021_202104{D}.nc'
    print(CMAQ_FileName)

    CMAQ_fh = Dataset(path+'/CCTM/'+CMAQ_FileName, mode='r')

    for V, T, U in zip(VAR_asl, VAR_asl_T, UNIT_asl): #zip = 
        print(V)

        temp = np.mean(CMAQ_fh.variables[V][:,0,:,:], axis=0)  # TSTEP, LAY, ROW, COL
        temp = np.squeeze(temp)

        ### Figure ###
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent(extent)

        pcolormesh = ax.pcolormesh(lon_np, lat_np, temp, cmap=custom_cmap)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax.add_feature(cfeature.STATES.with_scale('50m'))
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        # gl.xlabels_right = False  #<= 0.17
        # gl.ylabels_right = False  #<= 0.17
        
        gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1] + grid_lon_dist, grid_lon_dist))
        gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3] + grid_lat_dist, grid_lat_dist))
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        plt.colorbar(pcolormesh, ax=ax, orientation='vertical', pad=0.04)
        ax.set_title(f'CMAQ {T} on 2021-04-{D}', fontsize=15)
        
        plt.show()