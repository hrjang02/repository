import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import math
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair)


#wrf_file = Dataset('/data02/erkim/WRFLES_boseong/hourly/kst_hourly_'+time[j])
#wrf_file = Dataset('/data01/WRF_LES_2022/0929/copy/wrfout_d04_2022-09-28_15')
#wrf_file = Dataset('/data01/wrf_var/wrfout_d01_2022-07-21_18:00:00')
#wrf_file = Dataset('/home/erkim/03_WRFLES/wrfout_d04_2022-09-28_15')
wrf_file = Dataset('/data03/GMAP_FNL_run/WRF/202207_1km/wrfout_d06_2022-06-26_00:00:00')

ht = getvar(wrf_file, "z", timeidx=0)
ht = ht[0:16,65,67]
#ht = ht[0:75,22,199]
ter = getvar(wrf_file, "ter", timeidx=0)
#ter = ter[:,:,:]

print(ht)
