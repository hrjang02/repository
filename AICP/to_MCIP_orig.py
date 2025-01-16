
#%%
import pandas as pd
import xarray as xr
import numpy as np
import netCDF4 as nc
import pickle

Mvars = ['WSPD10']
Vvars = ['WS']
year = '2021'
start_time = pd.Timestamp('2020-12-25 01:00:00')

MCIP_orig = nc.Dataset(f'/home/hrjang2/METCRO2D_D02_{year}.nc', 'r+')

try:
    for Mvar, Vvar in zip(Mvars, Vvars):
        Mfile = MCIP_orig[Mvar][:, 0, :, :]

        Vfile = pd.read_csv(f'/data02/smkim/AICP/predicted/xgboost_model_predicted_{Vvar}.csv')
        Vfile['Time'] = pd.to_datetime(Vfile['Time'])

        Vfile['Row'] = Vfile['Unnamed: 0'].str.split('_').str[0].astype(int)
        Vfile['Col'] = Vfile['Unnamed: 0'].str.split('_').str[1].astype(int)
        Vfile['index'] = ((Vfile['Time'] - start_time) / pd.Timedelta(hours=1)).astype(int)
        if Vvar == 'WS':
            Vfile['0'] = Vfile['0'].where(Vfile['0'] >= 0, 0)
        for time in range(len(Mfile[:, 0, 0])):
            Vfile_time = Vfile[Vfile['index'] == time]
            for row in range(len(Mfile[0, :, 0])):
                Vfile_row = Vfile_time[Vfile_time['Row'] == row]
                for col in range(len(Mfile[0, 0, :])):
                    Vfile_col = Vfile_row[Vfile_row['Col'] == col]
                    if not Vfile_col.empty:
                        result = Vfile_col['0'].values
                        result = np.squeeze(result)
                        MCIP_orig[Mvar][time, 0, row, col] = np.float32(result)
                        print(f'time: {time}, row: {row}, col: {col}')
    MCIP_orig.sync()
finally:
    MCIP_orig.close()

# %%

# %%
'''
#승미 언니가 준 nc 만드는 코드

import pandas as pd
from netCDF4 import Dataset
from sat_toolbox import plot, geometry,cell_avg_2d
import numpy as np


car_filename_list = ['/data02/Ulsan_VOC/Data/Ulsan/20230808_Ulsan_1.xlsx',
                     '/data02/Ulsan_VOC/Data/Ulsan/20230808_Ulsan_2.xlsx',
                     '/data02/Ulsan_VOC/Data/Ulsan/20230814_Ulsan_1.xlsx',
                     '/data02/Ulsan_VOC/Data/Ulsan/20230814_Ulsan_2.xlsx',
                     '/data02/Ulsan_VOC/Data/Ulsan/20230816_Ulsan.xlsx']



for car_filename in car_filename_list:

    varname_list = ['측정 시간','benzene (ppb)','toluene (ppb)','xylenes + ethylbenzene (ppb)']

    list_2ds = []
    for varname in varname_list:
        array_2d,car_df = cell_avg_2d.avg_2d(car_filename,varname)
        list_2ds.append(array_2d)

    file_parts = car_filename.split('/')
    desired_part = file_parts[-1]  
    result_filename = desired_part.split('.')

    ext = Dataset('/home/smkim/work/cardoas/result/'+result_filename[-2]+'.nc', 'w', format='NETCDF4')

    ROW = ext.createDimension('ROW', 110)
    COL = ext.createDimension('COL', 110)
    
    SCANTIME = ext.createVariable('SCANTIME', 'u8',  ( 'ROW', 'COL'))
    BENZENE = ext.createVariable('BENZENE', 'f8',  ( 'ROW', 'COL'))
    TOL = ext.createVariable('TOL', 'f8',  ('ROW', 'COL'))
    XYL = ext.createVariable('XYL', 'f8',  ('ROW', 'COL'))
    
    SCANTIME[:] = list_2ds[0]
    BENZENE[:] = list_2ds[1]
    TOL[:] = list_2ds[2]
    XYL[:] = list_2ds[3]

    SCANTIME.description = 'Scantime (KST)'
    BENZENE.description = 'benzene (ppb)'
    TOL.description = 'toluene (ppb)'
    XYL.description = 'xylenes + ethylbenzene (ppb)'

    ext.close()

    result_df.to_csv(f"/data02/smkim/AICP/predicted/xgboost_model_predicted_{v}.csv")
'''
