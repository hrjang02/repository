#%%
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

year = '2021'
start_date = '2021-01-01'
end_date = '2021-12-31'
date = '2021'
APPL = '2021'
date_range = pd.date_range(start=start_date, end=end_date)
date_str_list = [date.strftime('%Y%m%d') for date in date_range]
start_Jdate = int(date_range[0].strftime('%Y%j'))
end_Jdate = int(date_range[-1].strftime('%Y%j'))
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
# %%
df_stn

df_stn.index = df_stn.index.tz_localize('Asia/Seoul')
df_stn.index = df_stn.index.tz_convert('UTC')
data_stn_UTC = df_stn.loc[start_date:end_date]
#%%
data_stn_UTC
# %%
data_stn_UTC.to_pickle(f'/home/smkim/0_code/validation/matched_pickle/cluster/O3_{start_date}_{end_date}.pkl')
# %%
