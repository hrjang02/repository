
#%%
import pandas as pd
import xarray as xr
import numpy as np
import netCDF4 as nc
from concurrent.futures import ProcessPoolExecutor

'''
# MCIP 원본/완성본 위치 확인하기(MCIP_path 확인하기)

domain = 'D02'
year = '2021'
months = [f'{month:02}' for month in range(1,13)]
Mvar = 'WSPD10'
var = 'predicted_ws'
try:
    for month in months:
        predicted_df = pd.read_pickle(f'/data07/WRF-ML/data/predicted/trial1/predicted_A_D02_{year}{month}')
        predicted_df['datetime'] = predicted_df['datetime'].dt.tz_localize(None)
        start_time = predicted_df['datetime'].min().replace(hour=1, minute=0, second=0, microsecond=0)
        predicted_df['time'] = ((predicted_df['datetime'] - start_time) / pd.Timedelta(hours=1)).astype(int)
        print(f"Start time for {year}{month}: {start_time}")

        # MCIP_orig = nc.Dataset(f'/data07/ULSAN_EHC/MCIP/{year}{month}/{domain}/METCRO2D_{domain}_{year}_EHC.nc')
        MCIP_orig = nc.Dataset(f'/home/hrjang2/METCRO2D_{domain}_EHC.nc', 'r+')
        Mfile = MCIP_orig[Mvar][:, 0, :, :]
        for time in range(len(Mfile[:, 0, 0])):
            file_time = predicted_df[predicted_df['time'] == time]
            for row in range(len(Mfile[0, :, 0])):
                Vfile_row = file_time[file_time['row'] == row]
                for col in range(len(Mfile[0, 0, :])):
                    Vfile_col = Vfile_row[Vfile_row['col'] == col]
                    if not Vfile_col.empty:
                        MLresult = Vfile_col[var].values
                        MLresult = np.squeeze(MLresult)
                        MCIP_orig[Mvar][time,0,row,col] = np.float32(MLresult)
                    print(f'time: {time}, row: {row}, col: {col}')
    MCIP_orig.sync()
except:
    print(f"Error in {year}{month}")
    pass
finally:
    MCIP_orig.close()
'''
# %%
#병렬 실행 코드 만들깅
def MCIP(year, month):
    domain = 'D02'
    Mvar = 'WSPD10'
    var = 'predicted_ws'

    predicted_df = pd.read_pickle(f'/data07/WRF-ML/data/predicted/trial1/predicted_A_D02_{year}{month}')
    predicted_df['datetime'] = predicted_df['datetime'].dt.tz_localize(None)
    start_time = predicted_df['datetime'].min().replace(hour=1, minute=0, second=0, microsecond=0)
    predicted_df['time'] = ((predicted_df['datetime'] - start_time) / pd.Timedelta(hours=1)).astype(int)
    print(f"Start time for {year}{month}: {start_time}")

    MCIP_orig = nc.Dataset(f'/home/hrjang2/METCRO2D_{domain}_EHC.nc', 'r+')
    # MCIP_orig = nc.Dataset(f'/data07/ULSAN_EHC/MCIP/{year}{month}/{domain}/METCRO2D_{domain}_{year}_EHC.nc')
    
    Mfile = MCIP_orig[Mvar][:, 0, :, :]
    for time in range(len(Mfile[:, 0, 0])):
        file_time = predicted_df[predicted_df['time'] == time]
        for row in range(len(Mfile[0, :, 0])):
            Vfile_row = file_time[file_time['row'] == row]
            for col in range(len(Mfile[0, 0, :])):
                Vfile_col = Vfile_row[Vfile_row['col'] == col]
                if not Vfile_col.empty:
                    MLresult = Vfile_col[var].values
                    MLresult = np.squeeze(MLresult)
                    try:
                        MCIP_orig[Mvar][time,0,row,col] = np.float32(MLresult)
                    except:
                        print(f'Error in {year}-{month}, time: {time}, row: {row}, col: {col}')
                        pass
    MCIP_orig.sync()
    MCIP_orig.close()
    print(f"Finished {year}{month}")

# 병렬 실행
if __name__ == '__main__':
    months = [f'{month:02}' for month in range(1, 13)]
    years = range(2021, 2022)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(MCIP, year, month) for year in years for month in months]
        for future in futures:
            future.result()

# %%
