#%%
import requests
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
import os
#%%
asos_path = '/data02/dongnam/met/ASOS/data/ASOS_stn_info.csv'
save_path = '/data02/dongnam/met/ASOS/raw'
api_key = '9eEYpmCCzAQKU3qUV5nLMejhomp0GufDpVIzIGL0JD4puo12TJ05L7gxt1tDgn09L4pIxCCjuEN4RXSrHjsDBw=='
api_url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'

asos_file = pd.read_csv(asos_path, encoding="EUC-KR")
stn_ids = asos_file['지점'].tolist()
#%%
def get_end_day(year, month):
    """주어진 연도와 월에 맞는 마지막 날짜 반환 (윤년 고려)"""
    if month in ['04', '06', '09', '11']:
        return '30'
    elif month == '02':
        year = int(year)
        return '29' if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else '28'
    else:
        return '31'
#%%
def download_asos(year):
    """
    연도별 ASOS 데이터를 월별로 받아 저장
    """
    year = str(year)

    months = ['01']
    for month in months:
        start_dt = f"{year}{month}01"
        end_dt = f"{year}{month}{get_end_day(year, month)}"

        all_data = []
        for stn in stn_ids:
            params = {
                'serviceKey': api_key,
                'pageNo': '1',
                'numOfRows': '999',
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'HR',
                'startDt': start_dt,
                'startHh': '00',
                'endDt': end_dt,
                'endHh': '23',
                'stnIds': stn
            }

            try:
                response = requests.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()
                items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                if items:
                    all_data.extend(items)
                else:
                    print(f"No data for station {stn} in {year}-{month}")
            except Exception as e:
                print(f"Error fetching station {stn} for {year}-{month}: {e}")
            

        if all_data:
            try:
                df = pd.DataFrame(all_data)
                df.to_csv(f'{save_path}/{year}{month}.csv', index=False, encoding="utf-8-sig")
                print(f"Saved {year}-{month} data")
            except Exception as e:
                print(f"Failed to save {year}-{month}: {e}")
        else:
            print(f"No data {year}-{month}")
#%%
if __name__ == "__main__":
    start_year = 2023
    end_year = 2023
    years = list(range(start_year, end_year + 1))

    with Pool(10) as p:  # 병렬 실행할 프로세스 수 설정
        p.map(download_asos, years)

#%%


