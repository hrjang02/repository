#%%
import pandas as pd
import numpy as np
from multiprocessing import Pool
from AQMS_TOOL import convert_24_to_00, POLLUTANT, DONGNAM
import matplotlib.pyplot as plt
#%%
# 변환 함수 정의 
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
    df[['SO2', 'O3', 'NO2']] *= 1000

    return df
#%%
_dict = {




#%%
# yearly trend
def plot_annual_trend(_df, _city, _pollutant, _include_type='all', _save_path='/home/hrjang2/0_code/pollutant/Yearly'):
    """ 
    특정 연도의 AQMS 데이터를 로드하고 전처리 

    Args:
        _df (pd.DataFrame): AQMS 데이터
        _city (str): 도시 이름 e.g. '부산'
        _pollutant (str): 오염물질 e.g. 'SO2'
        _include_type (str) : 'all' or '도시대기' (default: 'all')
        _path (str): 데이터 경로 
    """

    _df = _df[_df['city'] == _city] 

    if _include_type == 'all':
        _df = _df[_df['망'].str.contains('도시대기|도로변대기')]
    else:
        _df = _df[_df['망'].str.contains(f'{_include_type}')]

    yearly_avg = _df.groupby('Year').mean(numeric_only=True)[_pollutant]

    _city_name = DONGNAM(_city).city
    _pollutant_name = POLLUTANT(_pollutant).name
    _pollutant_unit = POLLUTANT(_pollutant).unit


    fig, ax = plt.subplots(figsize=(6, 3))
    yearly_avg.plot(ax=ax, marker = 'o', color='k')
    ax.set_title(f'{_city_name} {str(_pollutant_name)} Annual Trend', fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(f'{_pollutant_name} [{_pollutant_unit}]', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{_save_path}/{_city}_{_pollutant}_Yearly_trend.png')
    plt.show()
#%%
# 8-Hour Max Trend
# 'O3', 'CO'에 대해서만 적용
def plot_8hr_max_trend(_df, _city, _pollutant, _include_type='all', _save_path='/data02/dongnam/output_fig/pollutant/Yearly'):
    """
    특정 오염물질의 연도별 '일 최고 8시간 평균'의 연평균 추세
    
    Args:
        _df (pd.DataFrame): AQMS 데이터
        _city (str): 도시 이름 e.g. '부산'
        _pollutant (str): 오염물질 e.g. 'SO2'
        _include_type (str) : 'all' or '도시대기' (default: 'all')
        _save_path (str): 저장 경로 (default: '/data02/dongnam/output_fig/Yearly')
    """

    _df = _df[_df['city'] == _city]

    # '망' 필터링
    if _include_type == 'all':
        _df = _df[_df['망'].str.contains('도시대기|도로변대기')]
    else:
        _df = _df[_df['망'].str.contains(f'{_include_type}')]


    # 8시간 평균 계산 후, 일별 최고값 추출
    _df['측정일시'] = pd.to_datetime(_df['측정일시'])
    _df.set_index('측정일시', inplace=True)
    _df.sort_index(inplace=True) 

    _df['8hr_max'] = _df[_pollutant].rolling('8H', min_periods=8).mean()  # 8시간 평균 계산
    daily_max_8hr = _df.resample('D')['8hr_max'].max()  # 일별 최고 8시간 평균
    yearly_avg = daily_max_8hr.resample('Y').mean()
    
    _city_name = DONGNAM(_city).city
    _pollutant_name = POLLUTANT(_pollutant).name
    _pollutant_unit = POLLUTANT(_pollutant).unit

    fig, ax = plt.subplots(figsize=(6, 3))
    yearly_avg.plot(ax=ax, marker='o', color='k')
    ax.set_title(f'{_city_name} {_pollutant_name} 8-Hour Max Trend')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(f'{_pollutant_name} [{_pollutant_unit}]', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{_save_path}/{_city_name}_{_pollutant}_8hrMax_Yearly_Trend.png')
    plt.show()

#%%
# Monthly Trend Plot
def plot_monthly_trend(_year, _df, _city, _pollutant, _include_type='all', _save_path='/data02/dongnam/output_fig/pollutant/Monthly'):
    """
    특정 오염물질의 월별 평균 추세

    Args:
        _year (str): 기준연도
        _df (pd.DataFrame): AQMS 데이터
        _city (str): 도시 이름 e.g. '부산'
        _pollutant (str): 오염물질 e.g. 'SO2'
        _include_type (str) : 'all' or '도시대기' (default: 'all')
        _save_path (str): 저장 경로 (default: '/data02/dongnam/output_fig/Monthly')
    """

    plt.rc('font', family='NanumGothic') 
    plt.rcParams['axes.unicode_minus'] = False

    if _include_type == 'all':
        _df = _df[_df['망'].str.contains('도시대기|도로변대기')]
    else:
        _df = _df[_df['망'].str.contains(f'{_include_type}')]

    _df = _df[(_df['Year'] == str(_year)) & (_df['city'] == _city)].copy()
    _df['Month'] = _df['측정일시'].dt.month

    roadside_stations = _df.loc[_df['망'] == '도로변대기', '측정소명'].unique()
    monthly_avg = _df.groupby(['측정소명', 'Month'])[_pollutant].mean().unstack()

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    linestyles = ['-', '--', '-.', ':']

    _city_name = DONGNAM(_city).city
    _pollutant_name = POLLUTANT(_pollutant).name
    _pollutant_unit = POLLUTANT(_pollutant).unit

    fig, ax = plt.subplots(figsize=(7, 5))

    roadside_idx, general_idx = 0, 0

    for station in monthly_avg.index:
        if station not in roadside_stations:
            color_idx = general_idx % len(colors) 
            ax.plot(monthly_avg.columns, monthly_avg.loc[station], label=station, color=colors[color_idx], linewidth=1.5, alpha=1.0)
            general_idx += 1

    for station in monthly_avg.index:
        if station in roadside_stations:
            linestyle = linestyles[roadside_idx % len(linestyles)]
            ax.plot(monthly_avg.columns, monthly_avg.loc[station], label=station, linestyle=linestyle, color='black', linewidth=1.5, alpha=0.7)
            roadside_idx += 1

    avg_line = ax.plot(monthly_avg.columns, monthly_avg.mean(axis=0), color='red', linestyle='-', marker='o', linewidth=2, label='평균')
    ax.set_zorder(avg_line[0].get_zorder() + 1) 

    ax.set_title(f'{_city_name} {_pollutant_name} Monthly Trend', fontsize=12)
    ax.set_ylabel(f'{_pollutant_name} [{_pollutant_unit}]', fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    num_stations = len(monthly_avg.index)  
    legend_cols = min(6, num_stations)  
    legend_rows = int(np.ceil(num_stations / legend_cols))  
    bbox_y = -0.2 - (legend_rows - 1) * 0.1  

    ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, bbox_y), ncol=legend_cols, frameon=False)

    plt.tight_layout()
    #plt.savefig(f'{_save_path}/{_city_name}_{_pollutant}_monthly_trend.png', bbox_inches='tight', dpi=300)
    plt.show()
#%%
def plot_monthly_variation(_year, _df, _city, _pollutant, _save_path='/data02/dongnam/output_fig/pollutant/Monthly'):
    '''
    특정 오염물질의 월별 평균 추세
    Args:
        _year (str): 기준연도
        _df (pd.DataFrame): AQMS 데이터
        _city (str): 도시 이름 e.g. '부산'
        _pollutant (str): 오염물질 e.g. 'SO2'
        _save_path (str): 저장 경로 (default: '/data02/dongnam/output_fig/Monthly')
    '''

    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

    _df = _df[(_df['Year'] == str(_year)) & (_df['city'] == _city)]
    _df = _df[_df['망'].str.contains('도시대기')]
    

    _df['Month'] = _df['Month'].astype(int)
    monthly_avg = _df.groupby('Month').mean(numeric_only=True)[_pollutant]
    monthly_avg = monthly_avg.sort_index()


    _city_name = DONGNAM(_city).city
    _pollutant_name = POLLUTANT(_pollutant).name
    _pollutant_unit = POLLUTANT(_pollutant).unit



    fig, ax = plt.subplots(figsize=(6, 3))
    monthly_avg.plot(ax=ax, marker = 'o', color='k')
    ax.set_title(f'{_city_name} {_pollutant_name} Monthly Trend', fontsize=12)
    ax.set_ylabel(f'{_pollutant_name} [{_pollutant_unit}]', fontsize=10)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_xticks(monthly_avg.index)
    ax.set_xticklabels([f'{m}월' for m in monthly_avg.index])

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{_save_path}/{_city_name}_{_pollutant}_Monthly_variation.png', dpi=300)
    plt.show()

#%%##################################################
# 2019~2023년 DataFrame을 병렬로 처리
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


#%%##################################################
# Annual Trend Plot -> '도시대기'
pollutants = ['SO2', 'NO2', 'PM10', 'PM25', 'O3', 'CO']
dongnam_city = ['부산', '울산', '대구', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']

for city in dongnam_city:
    for pollutant in pollutants:
        print(f'{city} {pollutant} 연도별 평균')
        plot_annual_trend(AQMS, city, pollutant, _include_type='도시대기')


# %%##################################################
# 8-Hour Max Trend Plot -> '도시대기' // 'O3', 'CO'만 적용
pollutants = ['O3', 'CO']
dongnam_city = ['부산', '울산', '대구', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']

for city in dongnam_city:
    for pollutant in pollutants:
        print(f'{city} {pollutant} 8-Hour Max 연도별 평균')
        plot_8hr_max_trend(AQMS, city, pollutant, _include_type='도시대기')


# %%##################################################
# Monthly Trend Plot -> '도시대기','도로변대기'
pollutants = ['SO2', 'NO2', 'PM10', 'PM25', 'O3', 'CO']
dongnam_city = ['부산', '울산', '대구', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']
year = 2023

for city in dongnam_city:
    for pollutant in pollutants:
        print(f'{city} {pollutant} 월간 변화')
        plot_monthly_trend(year, AQMS, city, pollutant)

#%%
# %%##################################################
# Monthly Variation Plot -> '도시대기' // 'PM10', 'PM25'만 적용
pollutants = ['PM10', 'PM25']
dongnam_city = ['부산', '울산', '대구', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']

for city in dongnam_city:
    for pollutant in pollutants:
        print(f'{city} {pollutant} 월변화')
        plot_monthly_variation(year, AQMS, city, pollutant)

# %%
