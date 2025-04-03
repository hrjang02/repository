##################################################
# When you use this code, 
# Use 'select_region' function
# AQMS_data, AQMS_stn, AQMS_lon, AQMS_lat = select_region(2020, 'PM10', province='all', city = '영주')
##################################################
#%%
import pandas as pd
from glob import glob
#%%
class PROVINCE:
    def __init__(self, province):
        province_info = {
            '서울' : 'Seoul','부산': 'Busan','울산': 'Ulsan','대구': 'Daegu','대전' : 'Daejeon','광주' : 'Gwangju','인천' : 'Incheon',
            '세종' : 'Sejong','경남': 'Gyeongnam','경북': 'Gyeongbuk','전남': 'Jeonnam','전북': 'Jeonbuk','충남': 'Chungnam','충북': 'Chungbuk',
            '강원': 'Gangwon','경기': 'Gyeonggi','제주': 'Jeju'
        }
        self.province = province_info[province]
#%%
def load_AQMS_info(province='all'):
    '''Load AQMS information data for the province

    Args:
        province (str): province to load('all' or 'dongnam' or specific province like '강원')

    Return:
        AQMS_info (pd.DataFrame): AQMS information data for the province
    '''
    dongnam = ['부산', '울산', '대구', '경남', '경북']

    AQMS_info = pd.read_csv('/data01/Share/from_hrjang/AQMS_stn_info.csv')

    
    if province == 'dongnam':
        dongnam = ['Busan', 'Ulsan', 'Daegu', 'Gyeongnam', 'Gyeongbuk']
        for province in dongnam:
            AQMS_info = AQMS_info[AQMS_info['Province'] == province]
    else:
        AQMS_info = AQMS_info[AQMS_info['Province'] == province]

    return AQMS_info
#%%
def load_AQMS(year, pollutant, province):
    '''Load AQMS data for the year and province

    Args:
        year (int): year to load
        pollutant (str): pollutant to load
        province (str): province to load('all' or 'dongnam')

    Return:
        AQMS_data (pd.DataFrame): AQMS data for the year, province, and pollutant
    '''
    dongnam = ['Busan', 'Ulsan', 'Daegu', 'Gyeongnam', 'Gyeongbuk']

    AQMS_path = '/data01/Share/from_hrjang/AQMS/'
    AQMS_data = pd.DataFrame()

    if province == 'dongnam':
        for province in dongnam:
            file_path = f'{AQMS_path}{year}_{province}_{pollutant}.csv'
            files = glob(file_path)
            for file in files:
                province_data = pd.read_csv(file, index_col='측정일시')
                AQMS_data = pd.concat([AQMS_data, province_data], axis=1, ignore_index=False)
    else:
        file_path = f'{AQMS_path}{year}_*_{pollutant}.csv'
        files = glob(file_path) 
        for file in files:
            province_data = pd.read_csv(file, index_col='측정일시')
            AQMS_data = pd.concat([AQMS_data, province_data], axis=1, ignore_index=False)

    return AQMS_data
#%%
def select_region(year, pollutant, province='all', city = None):
    '''Select data for the province from the AQMS data

    Args:
        year (int): year to load
        pollutant (str): pollutant to load
        province (str): province to load(Default:'all' //or 'dongnam' or specific province like '경북')
            # 경남 고흥, 강원 고흥 구분 필요
        city (str): city to load(Default: None //or specific city like '영주')

    Return:
        AQMS_data (pd.DataFrame): AQMS data for the year, province, and pollutant
        AQMS_stn (list): AQMS station code for the province
        AQMS_lon (list): AQMS station longitude for the province
        AQMS_lat (list): AQMS station latitude for the province
    '''
    # Load AQMS information data
    AQMS_data = load_AQMS(year, pollutant, province)
    AQMS_info = load_AQMS_info(province)
    

    # Province filtering
    if province != 'all':
        AQMS_info = AQMS_info[AQMS_info['Province'] == province]

    # City filtering
    if city is not None:
        if isinstance(city, str):  
            city = [city]  # Ensure city is a list
        AQMS_info = AQMS_info[AQMS_info['City'].str[:2].isin(city)]

    # Extract station information
    AQMS_stn = AQMS_info['stn_code'].astype(str).to_list()
    AQMS_lon = AQMS_info['lon'].to_list()
    AQMS_lat = AQMS_info['lat'].to_list()

    if len(AQMS_stn) == 0:
        return AQMS_data, AQMS_stn, AQMS_lon, AQMS_lat  

    # Ensure AQMS_data only includes selected stations
    AQMS_data = AQMS_data.loc[:, AQMS_data.columns.str.contains('|'.join(AQMS_stn))]

    return AQMS_data, AQMS_stn, AQMS_lon, AQMS_lat

#%%
