#%%
import requests
import pandas as pd
#%%
def download_asos_data(year, asos_path, save_path, api_key):
    """
    ASOS 데이터를 연 단위로 다운로드해서 연-월.csv 형식으로 저장
    
    Parameters:
        year (str): 다운로드할 연도 (예: '2024')
        asos_path (str): ASOS 지점 정보 CSV 파일 경로
        save_path (str): 저장할 경로    
        api_key (str): 공공데이터 API 키
    """
    api_url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
    
    # ASOS 지점 목록 불러오기
    asos_file = pd.read_csv(asos_path, encoding="EUC-KR")
    stn_ids = asos_file['지점'].tolist()
    
    months = [str(i).zfill(2) for i in range(1, 13)]
    for month in months:
        start_dt = f"{year}{month}01"
        end_dt = f"{year}{month}{'30' if month in ['04','06','09','11'] else '28' if month == '02' else '31'}"
        all_data = []
        
        for stn in stn_ids:
            params = {
                'serviceKey': api_key,
                'pageNo': '1',
                'numOfRows': '750',
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
                all_data.extend(data['response']['body']['items']['item'])
            except:
                print(f"Error: {stn} {year}-{month}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(f'{save_path}/{year}{month}.csv', index=False, encoding="utf-8-sig")
            print(f"Saved {year}-{month} data")
        else:
            print(f"No data found for {year}-{month}")
#%%
# 사용 예시
download_asos_data(
    year='2005',
    asos_path='/data02/dongnam/met/ASOS/ASOS_stn_info.csv',
    save_path='/data02/dongnam/met/ASOS/raw',
    api_key='9eEYpmCCzAQKU3qUV5nLMejhomp0GufDpVIzIGL0JD4puo12TJ05L7gxt1tDgn09L4pIxCCjuEN4RXSrHjsDBw=='
)
#%%