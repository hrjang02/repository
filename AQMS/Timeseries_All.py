#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
sys.path.append('/home/hrjang2/0_code/AQMS')
from AQMS_TOOL import convert_24_to_00, POLLUTANT, CITY
#%%
# 데이터 불러오기
AQMS = []
for year in range(2019, 2024):
    df = pd.read_csv(f'/home/hrjang2/AQMS/data/{year}.csv', dtype={'Datetime': str}) 
    AQMS.append(df)  
AQMS = pd.concat(AQMS, ignore_index=True)


AQMS['측정일시'] = AQMS['측정일시'].astype(str)
AQMS['측정일시'] = AQMS['측정일시'].apply(convert_24_to_00)
AQMS['측정일시'] = pd.to_datetime(AQMS['측정일시'], format='%Y%m%d%H')

# 동남권 권역만 
dongnam = ['부산', '울산', '대구', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']
fAQMS = AQMS[AQMS['지역'].str.contains('|'.join(dongnam)) & ~AQMS['지역'].str.contains('강원 고성군') 
             & AQMS['망'].str.contains('도시대기')
].set_index('측정일시')

# year,month,day,DOY,province,city 컬럼 추가
fAQMS = fAQMS.assign(Year=fAQMS.index.year, Month=fAQMS.index.month, Day=fAQMS.index.day, DOY=fAQMS.index.dayofyear,
    province=fAQMS['지역'].str.split(' ').str[0],
    city=np.where(
        fAQMS['지역'].str.split(' ').str[0].isin(['경남', '경북']),
        fAQMS['지역'].str.split(' ').str[1].str[:2],
        fAQMS['지역'].str.split(' ').str[0])
        )
pollutants = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
fAQMS[['SO2', 'CO', 'O3', 'NO2']] *= 1000
#%%
def plot_station(start, end, frequency, output_dir = '/home/hrjang2/AQMS/figure/station', city = None):
    '''
    Parameters:
    - start: start time (YYYYMM, int)
    - end: end time (YYYYMM, int)
    - frequency: 'Yearly' or 'Monthly' (str)
    - output_dir: output directory path (str)
    - city: city name '부산'(str, default: None)
    '''
    group_key = {'Yearly': 'Year', 'Monthly': 'Month'}.get(frequency)
    
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
            standard = POLLUTANT(pollutant).standard
            city_name = CITY(fAQMS.loc[fAQMS['측정소코드'] == stnID, 'city'].unique()[0]).city

            if standard is not None and (y_values >= standard).any():
                ax.axhline(y=standard, color='r', linestyle='--', label=f'Standard: {standard} {unit}')
                print(f'{city_name} {stnID} {pollutant} {y_values[y_values > standard]}')

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
def plot_city(start, end, frequency, output_dir = '/home/hrjang2/AQMS/figure/city', city = None):
    '''
    Parameters:
    - start: start time (YYYYMM, int)
    - end: end time (YYYYMM, int)
    - group_by: 'city' (str)
    - frequency: 'Yearly' or 'Monthly' (str)
    - output_dir: output directory path (str)
    - city: city name '부산'(str, default: None)
    '''
    group_key = {'Yearly': 'Year', 'Monthly': 'Month'}.get(frequency)
    data = fAQMS[(fAQMS.index >= pd.to_datetime(start, format='%Y%m')) & (fAQMS.index <= pd.to_datetime(end, format='%Y%m'))]
    if city is not None:
        data = data[data['city'] == city]
    data = data.groupby(['city', group_key]).mean(numeric_only=True)
    groups = data.index.get_level_values('city').unique()

    for group in groups:
        stn_data = data.loc[group]
        x_range = stn_data.index.get_level_values(group_key)

        for pollutant in pollutants:
            fig, ax = plt.subplots(figsize=(5, 3))

            y_values = stn_data[pollutant]
            unit = POLLUTANT(pollutant).unit
            name = POLLUTANT(pollutant).name
            standard = POLLUTANT(pollutant).standard
            city_name = CITY(group).city

            if standard is not None and (y_values >= standard).any():
                ax.axhline(y=standard, color='r', linestyle='--', label=f'Standard: {standard} {unit}')
                print(f'{city_name} {pollutant} {y_values[y_values > standard]}')

            ax.plot(x_range, y_values, label=pollutant, marker='o', color = 'k')
            ax.set_xlabel(f'{group_key}', fontsize=12)
            ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
            ax.grid()

            ax.set_xticks(x_range)  
            ax.set_xticklabels(x_range, rotation=0, fontsize=10)

            fig.suptitle(f'{city_name} {name} {frequency} {start}-{end}', fontsize=15)
            fig.tight_layout()

            save_path = f'{output_dir}/{frequency}'
            os.makedirs(save_path, exist_ok=True)

            plt.savefig(f'{save_path}/{city_name}_{pollutant}_{frequency}.png')
            plt.close(fig)
# %%
def plot_city_yearmonth(start, end, frequency, output_dir = '/home/hrjang2/AQMS/figure/city', city = None):
    '''
    Parameters:
    - start: start time (YYYYMM, int)
    - end: end time (YYYYMM, int)
    - frequency: 'Yearly' or 'Monthly' (str)
    - output_dir: output directory path (str)
    - city: city name '부산'(str, default: None)
    '''
    group_key = {'Monthly': ['Year', 'Month']}.get(frequency)
    data = fAQMS[(fAQMS.index >= pd.to_datetime(start, format='%Y%m')) & (fAQMS.index <= pd.to_datetime(end, format='%Y%m'))]
    if city is not None:
        data = data[data['city'] == city]
    data = data.groupby(['city'] + group_key).mean(numeric_only=True)
    groups = data.index.get_level_values('city').unique()

    for group in groups:
        stn_data = data.loc[group]
        years = stn_data.index.get_level_values('Year').unique() 

        for pollutant in pollutants:
            fig, ax = plt.subplots(figsize=(5, 3))
            y_values = stn_data[pollutant]

            unit = POLLUTANT(pollutant).unit
            name = POLLUTANT(pollutant).name
            standard = POLLUTANT(pollutant).standard
            city_name = CITY(group).city

            if standard is not None and (y_values >= standard).any():
                ax.axhline(y=standard, color='r', linestyle='--', label=f'Standard: {standard} {unit}')
                print(f'{city_name} {pollutant} {y_values[y_values > standard]}')

            ax.plot(range(len(y_values)), y_values, marker='o', color='k', label=pollutant)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
            ax.grid()

            ax.set_xticks(range(0, len(y_values), 12)) 
            ax.set_xticklabels(years, fontsize=10)

            fig.suptitle(f'{city_name} {name} {year} Year{frequency}', fontsize=15)
            fig.tight_layout()

            save_path = f'{output_dir}/Year{frequency}'
            os.makedirs(save_path, exist_ok=True)

            plt.savefig(f'{save_path}/{city_name}_{pollutant}_Year{frequency}.png')
            plt.close(fig)
#%%
def plot_dongnam(start, end, frequency, output_dir = '/home/hrjang2/AQMS/figure/dongnam'):
    '''
    Parameters:
    - start: start time (YYYYMM, int)
    - end: end time (YYYYMM, int)
    - frequency: 'Yearly' or 'Monthly' (str)
    - output_dir: output directory path (str)
    '''
    group_key = {'Yearly': 'Year', 'Monthly': 'Month'}.get(frequency)
    data = fAQMS[(fAQMS.index >= pd.to_datetime(start, format='%Y%m')) & (fAQMS.index <= pd.to_datetime(end, format='%Y%m'))]
    data = data.groupby(group_key).mean(numeric_only=True)

    for pollutant in pollutants:
        fig, ax = plt.subplots(figsize=(5, 3))

        y_values = data[pollutant]
        unit = POLLUTANT(pollutant).unit
        name = POLLUTANT(pollutant).name
        standard = POLLUTANT(pollutant).standard

        if standard is not None and (y_values > standard).any():
            ax.axhline(y=standard, color='r', linestyle='--', label=f'Standard: {standard} {unit}')
            print(f'Dongnam {pollutant} {y_values[y_values > standard]}')
        
        x_range = data.index.get_level_values(group_key)

        ax.plot(x_range, y_values, label=pollutant, marker='o', color = 'k')
        ax.set_xlabel(f'{group_key}', fontsize=12)
        ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
        ax.grid()

        ax.set_xticks(x_range)  
        ax.set_xticklabels(x_range, rotation=0, fontsize=10)

        fig.suptitle(f'Dongnam {name} {frequency} {start}-{end}', fontsize=15)
        fig.tight_layout()

        save_path = f'{output_dir}'
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f'{save_path}/Dongnam_{pollutant}_{frequency}.png')
        plt.close(fig)
# %%
def plot_dongnam_yearmonth(start, end, frequency, output_dir = '/home/hrjang2/AQMS/figure/dongnam'):
    '''
    Parameters:
    - start: start time (YYYYMM, int)
    - end: end time (YYYYMM, int)
    - frequency: 'Yearly' or 'Monthly' (str)
    - output_dir: output directory path (str)
    '''
    group_key = {'Monthly': ['Year', 'Month']}.get(frequency)
    data = fAQMS[(fAQMS.index >= pd.to_datetime(start, format='%Y%m')) & (fAQMS.index <= pd.to_datetime(end, format='%Y%m'))]
    data = data.groupby(group_key).mean(numeric_only=True)
    years = data.index.get_level_values('Year').unique() 

    for pollutant in pollutants:
        fig, ax = plt.subplots(figsize=(5, 3))
        y_values = data[pollutant]

        unit = POLLUTANT(pollutant).unit
        name = POLLUTANT(pollutant).name
        standard = POLLUTANT(pollutant).standard

        if standard is not None and (y_values > standard).any():
            ax.axhline(y=standard, color='r', linestyle='--', label=f'Standard: {standard} {unit}')
            print(f'Dongnam {pollutant} {y_values[y_values > standard]}')
        ax.plot(range(len(y_values)), y_values, marker='o', color='k', label=pollutant)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
        ax.grid()

        ax.set_xticks(range(0, len(y_values), 12)) 
        ax.set_xticklabels(years, fontsize=10)

        fig.suptitle(f'Dongnam {name} Year{frequency} {start}-{end}', fontsize=15)
        fig.tight_layout()

        save_path = f'{output_dir}'
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f'{save_path}/Dongnam_{pollutant}_Year{frequency}.png')
        plt.close(fig)
# %%
def plot_doy(year, output_dir = '/home/hrjang2/AQMS/figure/city', city = None):
    '''
    Parameters:
    - year: year (int)
    - output_dir: output directory path (str)
    - city: city name '부산'(str, default: None)
    '''
    data = fAQMS[fAQMS['Year'] == year]
    if city is not None:
        data = data[data['city'] == city]
    data = data.groupby(['city', 'DOY']).mean(numeric_only=True)
    groups = data.index.get_level_values('city').unique()

    for group in groups:
        stn_data = data.loc[group]
        doys = stn_data.index.get_level_values('DOY').unique()

        for pollutant in pollutants:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            y_values = stn_data[pollutant]

            unit = POLLUTANT(pollutant).unit
            name = POLLUTANT(pollutant).name
            standard = POLLUTANT(pollutant).standard
            city_name = CITY(group).city

            if pollutant == 'PM25' and standard is not None and (y_values > standard).any():
                ax.axhline(y=standard, color='r', linestyle='--', label=f'Standard: {standard} {unit}')
                print(f'{city_name} {pollutant} {y_values[y_values > standard]}')

            ax.plot(doys, y_values, color='k', label=pollutant)
            ax.set_xlabel('DOY', fontsize=12)
            ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
            ax.grid()

            fig.suptitle(f'{city_name} {name} {year} DOY', fontsize=15)
            ax.autoscale(axis='x', tight=True)  
            fig.tight_layout()

            save_path = f'{output_dir}/DOY/{year}'
            os.makedirs(save_path, exist_ok=True)

            plt.savefig(f'{save_path}/{city_name}_{pollutant}_{year}_DOY.png')
            plt.close(fig)
# %%
