#%%
import pandas as pd
import xarray as xr
import numpy as np
import pyresample
from datetime import datetime
#%%
year = '2021'
start_date = '2021-10-01'
end_date = '2021-10-31'
date = '202110'
APPL = date
date_range = pd.date_range(start=start_date, end=end_date)
date_str_list = [date.strftime('%Y%m%d') for date in date_range]
start_Jdate = int(date_range[0].strftime('%Y%j'))
end_Jdate = int(date_range[-1].strftime('%Y%j'))


#%% select data from airkorea
var = 'ws'
csv_directory = '/home/smkim/materials/AWS/data'
df_stn = pd.read_csv(f'{csv_directory}/{year}_{var}.csv')
df_stn['tm'] = pd.to_datetime(df_stn['tm'])
df_stn.set_index('tm', inplace=True)

#%%
df_stn.index = df_stn.index.tz_localize('Asia/Seoul')
df_stn.index = df_stn.index.tz_convert('UTC')
data_stn_UTC = df_stn.loc[start_date:end_date]
#%%
data_stn_UTC
#%%
stn_nums = [col.split('_')[0] for col in data_stn_UTC.columns]
lon_stn = [float(col.split('_')[1]) for col in data_stn_UTC.columns]
lat_stn = [float(col.split('_')[2]) for col in data_stn_UTC.columns]

#%% extract data from WRF
grid_WRF = xr.open_dataset(f'/data07/dDN/mcip/D02_{year}/GRIDCRO2D_D02_{year}.nc')
# grid_WRF = xr.open_dataset(f'/home/smkim/CMAQ_dDN/data/mcip/{APPL}_D02_{year}/GRIDCRO2D_{APPL}_D02_{year}.nc')
lat_WRF = grid_WRF['LAT'].squeeze()
lon_WRF = grid_WRF['LON'].squeeze()
stn_def = pyresample.geometry.SwathDefinition(lons=lon_stn, lats=lat_stn)
WRF_def = pyresample.geometry.GridDefinition(lons=lon_WRF, lats=lat_WRF)


#%%
# MET_WRF = xr.open_dataset(f'/home/smkim/CMAQ_dDN/data/mcip/Nudging_D02_{year}/METCRO2D_D02_{year}.nc')
MET_WRF = xr.open_dataset(f'/data07/dDN/mcip/D02_{year}/METCRO2D_D02_{year}.nc')
# MET_WRF = xr.open_dataset(f'/home/smkim/CMAQ_dDN/data/mcip/D02_{year}/METCRO2D_D02_{year}.nc')
TFLAG_array = np.squeeze((MET_WRF.variables['TFLAG'][:,0,:]))
TFLAG_array.load()
start_TFLAG = np.where(TFLAG_array[:,0] == start_Jdate)[0][0]
end_TFLAG = np.where(TFLAG_array[:,0] == end_Jdate)[0][-1]

#%%
data_list = []
datetime_list = []
# data_reWRF_tmp = pd.DataFrame(columns=data_stn_UTC.columns)

for i in range(start_TFLAG, end_TFLAG + 1):
    print(i)
    datetime_Jint= int(TFLAG_array[i, 0].values)*1000000+int(TFLAG_array[i, -1].values)
    print(datetime_Jint)
    datetime_MCIP = datetime.strptime(str(datetime_Jint), "%Y%j%H%M%S")
    print(datetime_MCIP)
    # varData_WRF = np.squeeze((MET_WRF.variables['TEMP2'][i,:,:,:])).to_numpy()
    varData_WRF = np.squeeze((MET_WRF.variables['WSPD10'][i,:,:,:])).to_numpy()
    wf = lambda r: 1 - r/100000
    varData_re = pyresample.kd_tree.resample_custom(WRF_def, varData_WRF, stn_def, radius_of_influence=10000, neighbours=4,weight_funcs=wf, fill_value=np.nan)
    varData_re_df = pd.DataFrame(varData_re.reshape(1, -1), columns=data_stn_UTC.columns)
    data_list.append(varData_re_df)
    datetime_list.append(datetime_MCIP)
#%%
# Mstne DataFrame with proper time index(UTC)
data_reWRF = pd.concat(data_list, ignore_index=True)
data_reWRF.index = datetime_list
data_reWRF.index = data_reWRF.index.tz_localize('UTC')
#%%
fname = f'ML_WSPD_{start_date}_{end_date}.pkl'
print(f'FILENAME:{fname}')

#%%
data_reWRF.to_pickle(f'/home/smkim/0_code/validation/matched_pickle/{fname}')


# %%
