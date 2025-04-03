from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap


file_list = ['1001']


for i in range(0,np.shape(file_list)[0]):

    #data_path = '/home/erkim/python/10_KR_CH_static_201710_201803/results_statics/static_WS_'+file_list[i]+'.txt'
    data_path = '/home/erkim/03_WRFLES/NO2_v4/IUP_Ulsan_'+file_list[i]+'.csv'
    #data = np.loadtxt(data_path, dtype = float, skiprows = 1)
    data = np.loadtxt(data_path, delimiter=',', skiprows = 1, usecols=(2,3,4,5))

    print(data)

    lon = data[:,0]
    lat = data[:,1]

    RMS_999 = data[:,2]
   #RMS = np.ma.masked_where(MB_999<=0.002, RMS_999)

    NO2 = data[:,3]
    #NO2_999 = data[:,3]
    #NO2 = np.ma.masked_where(RMS_999 >=0.002, RMS_999)


    fig1 = plt.figure(figsize=(5, 5))


#    map = Basemap(width=20000, height=20000, rsphere=(6378137.00,6356752.3142), resolution='h',\
#              area_thresh=1000., projection='lcc', lat_0=35.54, lon_0=129.3) # downtown
    map = Basemap(width=15000, height=25000, rsphere=(6378137.00,6356752.3142), resolution='h',\
              area_thresh=1000., projection='lcc', lat_0=35.43, lon_0=129.34) # industrial

    map.drawcoastlines(linewidth=0.5)
    map.drawcountries(linewidth=0.5)
    map.drawparallels(np.arange(-90., 120., .1), labels=[1, 0, 0, 0])#np.arrange(min,max,)
    map.drawmeridians(np.arange(-180., 180., .1), labels=[0, 0, 0, 1])


    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.1, 'navy'), (0.2, 'blue'),\
           (0.3, 'cornflowerblue'), (0.4, 'turquoise'), (0.5, 'yellowgreen'), (0.6, 'yellow'), (0.7, 'orange'),\
           (0.8, 'orangered'), (0.9, 'red'), (1, 'darkred')], N=80)
    smap = map.scatter(lon, lat, s=10, latlon=True, c=NO2, cmap = cmap)
    #smap = map.scatter(lon, lat,s=15, latlon=True, c=corr, cmap = cmap, norm=norm, vmin = 0.1, vmax = 1)

    #legend
    plt.clim(0, 5e16)
    smaplegend = map.colorbar(smap,location='bottom',pad="10%")
    smaplegend.set_label('NO$_2$[${\mu}/m^3$]')
    plt.title('NO$_2$ concentration (10/01)')
    plt.tight_layout()
#    fig1.savefig('./fig_statics/WS_corr_'+file_list[i]+'.png',dpi=200)
    plt.show()
