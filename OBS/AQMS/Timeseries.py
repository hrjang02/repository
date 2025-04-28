#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
sys.path.append('/data02/dongnam/code')
from AQMS_TOOL import convert_24_to_00, POLLUTANT, DONGNAM, PROVINCE
from multiprocessing import Pool
#%%
def load_and_preprocess(year, path='/data02/dongnam/data'):
    """ 특정 연도의 AQMS 데이터를 로드하고 전처리 
    Args:
        year (int): 연도
        path (str): 데이터 경로 (default: '/data02/dongnam/data')
        Returns:
            pd.DataFrame: 전처리된 AQMS 데이터
    """
    df = pd.read_csv(f'{path}/{year}.csv', dtype={'Datetime': str})
    
    df['측정일시'] = df['측정일시'].astype(str).apply(convert_24_to_00)
    df['측정일시'] = pd.to_datetime(df['측정일시'], format='%Y%m%d%H')

    # 동남권 권역 필터링
    dongnam = ['부산', '울산', '대구', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']
    df = df[df['지역'].str.contains('|'.join(dongnam)) & ~df['지역'].str.contains('강원 고성군')]    
    
    df = df.set_index('측정일시')
    df = df.assign(Year=df.index.year.astype(str), Month=df.index.month.astype(str), province=df['지역'].str.split().str[0],
        city=np.where(df['지역'].str.split().str[0].isin(['경남', '경북']),df['지역'].str.split().str[1].str[:2],df['지역'].str.split().str[0]
        ))
    df['측정일시'] = df.index 
    df[['SO2', 'O3', 'NO2' , 'CO']] *= 1000

    return df
#%%
startyear = 2019
endyear = 2023

if __name__ == "__main__":
    years = range(startyear, endyear + 1)
    with Pool(10) as p:  # 4개의 프로세스로 병렬 실행
        results = p.map(load_and_preprocess, years)
    AQMS = pd.concat(results, ignore_index=True)
    AQMS = AQMS[(AQMS['Year'] >= str(startyear)) & (AQMS['Year'] <= str(endyear))]
    AQMS = AQMS.reset_index(drop=True)
    print("병렬 처리 끝!")
#%%
fAQMS = AQMS.copy()
pollutants = ['SO2', 'O3', 'NO2', 'CO', 'PM10', 'PM25']
#%%
def plot_station(start, end, frequency, output_dir = '/data02/dongnam/data/output_fig/hrjang_fig/', city = None):
    '''
    Parameters:
    - start: start time (YYYYMM, int)
    - end: end time (YYYYMM, int)
    - frequency: 'Annual' or 'Monthly' (str)
    - output_dir: output directory path (str)
    - city: city name '부산'(str, default: None)
    '''
    group_key = {'Annual': 'Year', 'Monthly': 'Month'}.get(frequency)
    
    data = fAQMS[(fAQMS.index >= pd.to_datetime(start, format='%Y%m')) & (fAQMS.index <= pd.to_datetime(end, format='%Y%m'))]
    if city is not None:
        data = data[data['city'] == city]
    data = data.groupby(['측정소코드', group_key]).mean(numeric_only=True)
    stnIDs = data.index.get_level_values('측정소코드').unique()

    for stnID in stnIDs:
        stn_data = data.loc[stnID]
        x_range = stn_data.index.get_level_values(group_key)
        for pollutant in pollutants:
            fig, ax = plt.subplots(figsize=(5, 3))

            y_values = stn_data[pollutant]
            unit = POLLUTANT(pollutant).unit
            name = POLLUTANT(pollutant).name
            city_name = DONGNAM(fAQMS.loc[fAQMS['측정소코드'] == stnID, 'city'].unique()[0]).city

            ax.plot(x_range, y_values, label=pollutant, marker='o', color = 'k')
            ax.set_xlabel(f'{group_key}', fontsize=12)
            ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
            ax.grid()

            ax.set_xticks(x_range)  
            ax.set_xticklabels(x_range, rotation=0, fontsize=10)

            fig.suptitle(f'{city_name} {stnID} {name} {frequency} {start}-{end}', fontsize=15)
            fig.tight_layout()

            save_path = f'{output_dir}/{frequency}'
            os.makedirs(save_path, exist_ok=True)

            plt.savefig(f'{save_path}/{city_name}_{stnID}_{pollutant}_{frequency}.png')
            plt.close(fig)
#%%
def plot_annual_trend(data, area_name=None, area_type='Dongnam', save_path='/home/hrjang2/0_code/yearly'):
    '''
    Parameters:
    - data: DataFrame (columns: ['Year', 'city', 'province', pollutant...])
    - area_type: 'Dongnam', 'city', or 'province' (str)
    - area_name: name of city or province, required if area_type is not 'Dongnam' (default: None)
    - save_path: directory to save figures (str)
    '''

    if area_type == 'city' and area_name:
        areas = [area_name]
        key = 'city'
        grouped = data[data['city'] == area_name].groupby('Year').mean(numeric_only=True)

    elif area_type == 'province' and area_name:
        areas = [area_name]
        key = 'province'
        grouped = data[data['province'] == area_name].groupby('Year').mean(numeric_only=True)

    elif area_type == 'Dongnam':
        areas = ['Dongnam']
        grouped = data.groupby('Year').mean(numeric_only=True)

    for area in areas:
        if area_type == 'Dongnam':
            area_data = grouped
            title_prefix = 'Dongnam'
            filename_prefix = 'Dongnam'
        elif area_type in ['city', 'province'] and area_name is None:
            area_data = grouped.loc[area]
            title_prefix = DONGNAM(area).city if area_type == 'city' else PROVINCE(area)
            filename_prefix = title_prefix
        else:
            area_data = grouped
            title_prefix = DONGNAM(area).city if area_type == 'city' else PROVINCE(area)
            filename_prefix = title_prefix

        x_range = area_data.index

        for pollutant in pollutants:
            fig, ax = plt.subplots(figsize=(6, 3))
            y_values = area_data[pollutant]

            unit = POLLUTANT(pollutant).unit
            name = POLLUTANT(pollutant).name

            ax.plot(x_range, y_values, marker='o', color='k', label=pollutant)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
            ax.set_xticks(x_range)
            ax.set_xticklabels(x_range, rotation=0, fontsize=10)
            ax.grid()

            fig.suptitle(f'{title_prefix} {name} Annual Trend', fontsize=15)
            fig.tight_layout()

            plt.savefig(f'{save_path}/{filename_prefix}_{pollutant}_Yearly.png')
            plt.close(fig)
#%%
############################################################
# 도시별
citys = fAQMS['city'].unique()
for city in citys:
    plot_annual_trend(fAQMS, city, 'city')

# 전체 평균
plot_annual_trend(fAQMS, 'Dongnam')

# 시도별
provinces = fAQMS['province'].unique()
for province in provinces:
    plot_annual_trend(fAQMS, province, 'province')

############################################################
# %%
def plot_yearmonth_trend(data, frequency, save_path='/home/hrjang2/0_code/yearmonth', groupby_key=None):
    '''
    Parameters:
    - data: DataFrame with pollution data
    - frequency: 'Monthly' or 'Annual'
    - save_path: Directory to save output figures
    - groupby_key: None (whole data), 'city', or 'province'
    '''
    group_key = {'Monthly': ['Year', 'Month']}.get(frequency)
    
    if groupby_key is not None:
        data = data.groupby([groupby_key] + group_key).mean(numeric_only=True)
        groups = data.index.get_level_values(groupby_key).unique()
    else:
        data = data.groupby(group_key).mean(numeric_only=True)
        groups = [None]  # 하나의 그룹만 존재

    for group in groups:
        if groupby_key is not None:
            stn_data = data.loc[group]
        else:
            stn_data = data
        years = stn_data.index.get_level_values('Year').unique()

        for pollutant in pollutants:
            fig, ax = plt.subplots(figsize=(8, 3))
            y_values = stn_data[pollutant]

            unit = POLLUTANT(pollutant).unit
            pol_name = POLLUTANT(pollutant).name

            if groupby_key == 'city':
                title_name = DONGNAM(group).city
            elif groupby_key == 'province':
                title_name = PROVINCE(group)
            else:
                title_name = 'Dongnam'

            ax.plot(range(len(y_values)), y_values, marker = 'o', color='k', label=pollutant)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(f'{pol_name} [{unit}]', fontsize=12)
            ax.grid()

            ax.set_xticks(range(0, len(y_values), 12)) 
            ax.set_xticklabels(years, fontsize=10)

            fig.suptitle(f'{title_name} {pol_name} {frequency} Trend', fontsize=15)
            fig.tight_layout()

            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/{title_name}_{pollutant}_Year{frequency}.png')
            plt.close(fig)
#%%
############################################################
# 도시별
plot_yearmonth_trend(fAQMS, 'Monthly', groupby_key='city')

# 전체 평균
plot_yearmonth_trend(fAQMS, 'Monthly', groupby_key=None)

# 시도별
plot_yearmonth_trend(fAQMS, 'Monthly', groupby_key='province')

############################################################
# %%
