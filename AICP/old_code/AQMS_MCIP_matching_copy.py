#%%
import pandas as pd
import xarray as xr
import numpy as np
import pyresample
from datetime import datetime
#%%
year = '2023'
start_date = '2023-01-01'
end_date = '2023-12-31'
date = '2023'
APPL = '2023'
date_range = pd.date_range(start=start_date, end=end_date)
date_str_list = [date.strftime('%Y%m%d') for date in date_range]
start_Jdate = int(date_range[0].strftime('%Y%j'))
end_Jdate = int(date_range[-1].strftime('%Y%j'))

#%% Select station
# stn_df_all = pd.read_csv('/home/smkim/work/KRCN/refer/Surface_stn.csv')
# stn_df = stn_df_all[~stn_df_all['STN'].str.endswith('A')]
# lat_stn = stn_df['LAT']
# lon_stn = stn_df['LON']
# id_stn = stn_df['STN']

#%% select data from stn#%% Select station
stn_df_all = pd.read_csv('/home/smkim/work/KRCN/refer/Surface_stn.csv')
stn_df = stn_df_all[~stn_df_all['STN'].str.endswith('A')]
lat_stn = stn_df['LAT']
lon_stn = stn_df['LON']
id_stn = stn_df['STN']

var = 'O3'
csv_directory = '/home/smkim/materials/pollutant/data'
df_stn = pd.read_csv(f'{csv_directory}/{year}_{var}.csv',usecols=lambda column: column != 'Unnamed: 0')
df_stn['측정일시'] = pd.to_datetime(df_stn['측정일시'])
df_stn.set_index('측정일시', inplace=True)

header_str = list(id_stn)
Korea_columns = [col for col in df_stn.columns if any(col.startswith(stnid) for stnid in header_str)]

data_stn_tmp = df_stn[Korea_columns]
# data_stn_tmp = df_stn

#%%
df_stn

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
# grid_WRF = xr.open_dataset(f'/data07/dDN/mcip/D02_{year}/GRIDCRO2D_D02_{year}.nc')
grid_WRF = xr.open_dataset(f'/home/smkim/CMAQ_dDN/data/mcip/D02_{year}/GRIDCRO2D_D02_{year}.nc')
lat_WRF = grid_WRF['LAT'].squeeze()
lon_WRF = grid_WRF['LON'].squeeze()
stn_def = pyresample.geometry.SwathDefinition(lons=lon_stn, lats=lat_stn)
WRF_def = pyresample.geometry.GridDefinition(lons=lon_WRF, lats=lat_WRF)


#%%
# MET_WRF = xr.open_dataset(f'/home/smkim/CMAQ_dDN/data/mcip/Nudging_D02_{year}/METCRO2D_D02_{year}.nc')
MET_WRF = xr.open_dataset(f'/home/smkim/CMAQ_dDN/data/mcip/D02_{year}/METCRO2D_D02_{year}.nc')
# MET_WRF = xr.open_dataset(f'/home/smkim/CMAQ_dDN/data/mcip/D02_{year}/METCRO2D_D02_{year}.nc')
TFLAG_array = np.squeeze((MET_WRF.variables['TFLAG'][:,0,:]))
TFLAG_array.load()
start_TFLAG = np.where(TFLAG_array[:,0] == start_Jdate)[0][0]
end_TFLAG = np.where(TFLAG_array[:,0] == end_Jdate)[0][-1]

#%%
def matching(varname):
    data_list = []
    datetime_list = []
    # data_reWRF_tmp = pd.DataFrame(columns=data_stn_UTC.columns)

    for i in range(start_TFLAG, end_TFLAG + 1):
        print(i)
        datetime_Jint= int(TFLAG_array[i, 0].values)*1000000+int(TFLAG_array[i, -1].values)
        print(datetime_Jint)
        datetime_MCIP = datetime.strptime(str(datetime_Jint), "%Y%j%H%M%S")
        print(datetime_MCIP)
        varData_WRF = np.squeeze((MET_WRF.variables[varname][i,:,:,:])).to_numpy()
        wf = lambda r: 1 - r/100000
        varData_re = pyresample.kd_tree.resample_custom(WRF_def, varData_WRF, stn_def, radius_of_influence=10000, neighbours=4,weight_funcs=wf, fill_value=np.nan)
        varData_re_df = pd.DataFrame(varData_re.reshape(1, -1), columns=data_stn_UTC.columns)
        data_list.append(varData_re_df)
        datetime_list.append(datetime_MCIP)
    return data_list, datetime_list
#%%
# Mstne DataFrame with proper time index(UTC)
for varname in list(MET_WRF.variables)[28:]:
    try:
        data_list, datetime_list = matching(varname)
        data_reWRF = pd.concat(data_list, ignore_index=True)
    except: 
        print(f'{varname} is not available')
        continue
    data_reWRF.index = datetime_list
    data_reWRF.index = data_reWRF.index.tz_localize('UTC')
    print(f'FILENAME:{varname}_{start_date}_{end_date}.pkl')
    data_reWRF.to_pickle(f'/home/smkim/0_code/validation/matched_pickle/cluster/{varname}_{start_date}_{end_date}.pkl')


# %%
