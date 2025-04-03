############################################################################
# Pre-process Air Quality Monitoring System (AQMS) data
# Designed for Airkorea confirmed dataset
# Programmed under CentOS7, Python 3.10
# Written by Hyoran Jang
############################################################################
#%%
import numpy as np
import pandas as pd

# datetime 만들어요 (24시 처리 함수)~~~~~~~~~~~~~ 
def convert_24_to_00(datetime):
    """
    Convert 24-hour datetime to 00-hour datetime

    input: datetime (str) - datetime in format of '%Y%m%d24'
    output: datetime (str) - datetime in format of '%Y%m%d%00'
    """

    if datetime.endswith("24"):
        date_part = datetime[:-2]
        next_date = pd.to_datetime(date_part, format='%Y%m%d') + pd.Timedelta(days=1)
        return next_date.strftime('%Y%m%d00')
    return datetime

def call_aqms(year, month, file_path='/home/hrjang2/AQMS/data', region='all'):
    from glob import glob
    """
    Call list of files in the given year and month

    input: year (str) - year to call files (format: 'YYYY')
           month (str) - month to call files (format: 'MM' or 'all')
           file_path (str) - path to the data files
    output: files (list) - list of files in the given year and month
    """

    if type(year) == str:
        year_list = [year]

    if month == 'all':
        files = glob(f'{file_path}/{year}/*.csv')
        count = 0
        for file in files:
            df_month = pd.read_csv(file, dtype={'Datetime': str})
            if count == 0:
                df = df_month
                count += 1
            else:
                df = pd.concat([df, df_month], ignore_index=True)
                
    else:
        files = glob(f'{file_path}/{year}/{year}{month}.csv')
        df = pd.read_csv(files[0], dtype={'Datetime': str})
    
    df['Datetime'] = df['Datetime'].apply(convert_24_to_00)
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H')

    df.drop(['망', 'stn_name', '주소'], axis=1, inplace=True)

    if region == 'dongnam':
        regions = ['부산 ', '울산 ', '대구 ', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']
        df = df[df['Region'].str.contains('|'.join(regions)) & ~df['Region'].str.contains('강원 고성군')]
        df['province'] = df['Region'].str.split(' ').str[0]
        df['city'] = np.where(df['province'].isin(['경북']), df['Region'].str.split(' ').str[1].str[:2], df['Region'].str.split(' ').str[0])

    return df

class FREQ_GROUPBY:
    def __init__(self, data):
        self.yearly = data.groupby(['Year']).mean(numeric_only=True)
        self.monthly = data.groupby(['Month']).mean(numeric_only=True)
        self.daily = data.groupby(['Day']).mean(numeric_only=True)
