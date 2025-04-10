#%%
from urllib.request import urlopen, Request
from urllib.parse import urlencode, unquote, quote_plus
import xml.etree.ElementTree as ET
import csv
import requests
import pandas as pd
import json
import requests



#%%
stn_list = []

with open("/home/smkim/materials/ASOS/unique_ASOS_stn.txt", "r") as file:
    for line in file:
        stn_list.append(line.strip())

print(stn_list)
#%%
year = '2024'
months = [str(i).zfill(2) for i in range(11, 13)]

failed_cases = []

for month in months:
    df = pd.DataFrame()
    startDt = year+month+'01'
    if month in ['04','06','09','11']:
        endDt = year+month+'30'
    elif month in ['02']:
        endDt = year+month+'28'
    else:
        endDt = year+month+'31'
    
    
    for stn in stn_list:
        
        url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
        params ={'serviceKey' : 'j5nsI2v5/OskzuZBbLFAYx7NoAC1bY/j72C+ANVCUjXEKIdTz+Pc8+sS5HsaVR2I1xeK2xdmAy+poVZ5aRt+mg==', 
                'pageNo' : '1', 
                'numOfRows' : '750', 
                'dataType' : 'JSON', 
                'dataCd' : 'ASOS', 
                'dateCd' : 'HR', 
                'startDt' : startDt, 
                'startHh' : '00', 
                'endDt' : endDt, 
                'endHh' : '23', 
                'stnIds' : stn 
                }
        max_retries = 4
        for _ in range(max_retries):
            try:
                response = requests.get(url, params=params)
                data = response.json()
                num_data = data['response']['body']['totalCount']
                list_2d = data['response']['body']['items']['item']
                tmp_df = pd.DataFrame(list_2d)
                df = pd.concat([df,tmp_df],axis=0)
                
                df.to_csv(f'/home/smkim/materials/ASOS/raw/{year}{month}.csv')
                print('Success to save for', stn, year, month)
                
                # 성공했으므로 반복문 종료
                break
                
            except Exception as e:
                print(f"Error: {year}{month} at {stn}", e)
                
                if _ == max_retries - 1:
                    save_str = f'{stn} {year} {month}'
                    failed_cases.append(save_str)
                    print(f"Failed to save for {stn} after {max_retries} retries")
                continue
# %%
