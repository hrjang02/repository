#%%
import pandas as pd
import xarray as xr
import numpy as np
import pyresample

#%%
year = '2021'
start_date = '2021-01-01'
end_date = '2021-12-31'
date = '202110'
APPL = 'RIST'
date_range = pd.date_range(start=start_date, end=end_date)
date_str_list = [date.strftime('%Y%m%d') for date in date_range]

#%% Select station
stn_df_all = pd.read_csv('/home/smkim/work/KRCN/refer/Surface_stn.csv')
stn_df = stn_df_all[~stn_df_all['STN'].str.endswith('A')]
lat_stn = stn_df['LAT']
lon_stn = stn_df['LON']
id_stn = stn_df['STN']

#%% select data from airkorea
var = 'O3'
csv_directory = '/home/smkim/materials/pollutant/data'
df_AK = pd.read_csv(f'{csv_directory}/{year}_{var}.csv',usecols=lambda column: column != 'Unnamed: 0')
df_AK['측정일시'] = pd.to_datetime(df_AK['측정일시'])
df_AK.set_index('측정일시', inplace=True)

header_str = list(id_stn)
Korea_columns = [col for col in df_AK.columns if any(col.startswith(stnid) for stnid in header_str)]

data_AK_tmp = df_AK[Korea_columns]
# data_AK_tmp = df_AK
data_AK_tmp.index = data_AK_tmp.index.tz_localize('Asia/Seoul')
data_AK_tmp.index = data_AK_tmp.index.tz_convert('UTC')
data_AK_UTC = data_AK_tmp.loc[start_date:end_date]

#%%
stn_nums = [col.split('_')[0] for col in data_AK_UTC.columns]
lon_stn = [float(col.split('_')[1]) for col in data_AK_UTC.columns]
lat_stn = [float(col.split('_')[2]) for col in data_AK_UTC.columns]

#%% extract data from CMAQ
grid_CMAQ = xr.open_dataset(f'/data05/RIST/mcip/KLC_30/202101/d02/GRIDCRO2D_{year}01_d02.nc')
lat_CMAQ = grid_CMAQ['LAT'].squeeze()
lon_CMAQ = grid_CMAQ['LON'].squeeze()
AK_def = pyresample.geometry.SwathDefinition(lons=lon_stn, lats=lat_stn)
CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)
#%%
data_list = []
cmaq_datetime_list = []
data_reCMAQ = pd.DataFrame(columns=data_AK_UTC.columns)

ACONC_CMAQ = xr.open_dataset(f'/data05/RIST/combine/base/{date}/d02/COMBINE_ACONC_v54_intel_{date}')
daily_CONC_CMAQ = ACONC_CMAQ.variables[var]
daily_TFLAG_CMAQ = ACONC_CMAQ.variables['TFLAG'][:, 0, :].values

for i in range(daily_CONC_CMAQ.shape[0]):    #shape[0] means first dimension size of that xarray
    ozone_CMAQ = np.squeeze((daily_CONC_CMAQ[i,:,:,:])).to_numpy()
    ozone_re = pyresample.kd_tree.resample_nearest(CMAQ_def, ozone_CMAQ, AK_def, radius_of_influence=10000, fill_value=np.nan)
    ozone_re_df = pd.DataFrame(ozone_re.reshape(1, -1), columns=data_AK_UTC.columns)
    datetime_CMAQ = pd.to_datetime((np.int64(daily_TFLAG_CMAQ[i, 0]) * 1000000 + daily_TFLAG_CMAQ[i, 1]).astype(str), format='%Y%j%H%M%S')
    data_list.append(ozone_re_df)
    cmaq_datetime_list.append(datetime_CMAQ)
    print(datetime_CMAQ)
#%%
# Make DataFrame with proper time index(UTC)
data_reCMAQ = pd.concat(data_list, ignore_index=True)
data_reCMAQ.index = cmaq_datetime_list
data_reCMAQ.index = data_reCMAQ.index.tz_localize('UTC')
#%%
# data_reCMAQ.to_pickle(f'/home/smkim/0_code/validation/matched_pickle/{APPL}_{start_date}_{end_date}.pkl')

# %%
