#%%
import pandas as pd
import numpy as np
import os
from glob import glob


# 파일 경로 설정
year = '2024'
sat_path = f'/data02/Airkorea/{year}/'
AK_csv = sat_path + f'{year}0[4-6]*.csv'
file_list = glob(AK_csv)
valid_files = [file for file in file_list if os.path.getsize(file) > 0]
AK_list = sorted(valid_files, key=lambda x: int(os.path.basename(x)[:10]))


AK_df = []

for file in AK_list:
    try:
        df = pd.read_csv(file)
        
        dateframe = pd.DataFrame({
            'lat': df.iloc[:, 1].round(3),
            'lon': df.iloc[:, 2].round(3),
            'datetime': df.iloc[:, 17],
            #'SO2': df.iloc[:, 4],
            #'CO': df.iloc[:, 11],
            'O3': df.iloc[:, 12],
            #'NO2': df.iloc[:, 10],
            #'PM10': df.iloc[:, 22],
            #'PM25': df.iloc[:, 14]
        })

        AK_df.append(dateframe)
    
    except pd.errors.EmptyDataError:
        print(f"EmptyDataError: No data in file {file}")

AK = pd.concat(AK_df, ignore_index=True)
print(AK)

#%%파일 저장
csv_filename = f'/home/hrjang/practice/AQMS_{year}_O3.csv'  
AK.to_csv(csv_filename, index=False)
#%%
import pandas as pd
import os
from glob import glob

#stn_info_file = '/data02/Airkorea/AQMS_stn_info.csv'
#stn_info_df = pd.read_csv(stn_info_file)
df_info = pd.read_csv('/home/intern/data/pollutant/2021_NO2.csv')
df_info['측정일시'] = pd.to_datetime(df_info['측정일시'])

obs_stn = df_info.columns[2:].str.split('_').str[0]
obs_lon = df_info.columns[2:].str.split('_').str[1].astype(float)
obs_lat = df_info.columns[2:].str.split('_').str[2].astype(float)
stn_info_df = pd.DataFrame({'stdID' : obs_stn,'lat' : obs_lat,'lon': obs_lon})

sat_path = '/data02/Airkorea/2024/'
AK_csv = sat_path + '20240[4-6]*.csv'


file_list = glob(AK_csv)
valid_files = [file for file in file_list if os.path.getsize(file) > 0]
AK_list = sorted(valid_files, key=lambda x: int(os.path.basename(x)[:10]))


AK_df = []
for file in AK_list:
    try:
        df = pd.read_csv(file)
        
        dateframe = pd.DataFrame({
            'datetime': df.iloc[:, 17],
            #'SO2': df.iloc[:, 4],
            #'CO': df.iloc[:, 11],
            'O3': df.iloc[:, 12],
            #'NO2': df.iloc[:, 10],
            #'PM10': df.iloc[:, 22],
            #'PM25': df.iloc[:, 14],
            'lat': df.iloc[:, 1],  
            'lon': df.iloc[:, 2]  
        })
        
        AK_df.append(dateframe)
    
    except pd.errors.EmptyDataError:
        print(f"EmptyDataError: No data in file {file}")

AK = pd.concat(AK_df, ignore_index=True)

AK_with_stnID = pd.merge(AK, stn_info_df,
                         left_on=['lat', 'lon'],
                         right_on=['lat', 'lon'], how='outer')

print(AK_with_stnID)
# %%
