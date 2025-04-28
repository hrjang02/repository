#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress
from glob import glob
#%%
ASOS_path = '/data02/dongnam/met/ASOS/raw/*'
files = glob(ASOS_path)
#%%
# ASOS_stn_info.csv 파일에서 지점명과 지점번호를 불러오기
stn_info = pd.read_csv('/data02/dongnam/met/ASOS/data/dongnam_ASOS_info.csv', encoding='euc-kr')
stnID = stn_info['지점'].unique().tolist()
stnNm = stn_info['지점명'].unique().tolist()
#%%
def load_wind(_files):
    '''
    Load wind data from CSV files and combine them into a single DataFrame.

    Args:
    _files (list) : List of file paths to CSV files

    Returns:
    DataFrame : Combined DataFrame containing wind data
    '''
    data_list = []
    for file in _files:
        df = pd.read_csv(file, usecols=['KST', 'stnNm', 'ws'])

        df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%d %H:%M', errors='coerce')
        df['ws'] = pd.to_numeric(df['ws'], errors='coerce')
        df['년'] = df['tm'].dt.year
        df['월'] = df['tm'].dt.month
        
        data_list.append(df)
        
    combined_df = pd.concat(data_list, ignore_index=True)
    # combined_df = combined_df.drop_duplicates()

    return combined_df
#%%
def select_data(_df, _city, _month):
    '''
    Filter the wind data for a specific city and month.

    Args:
    _df (DataFrame) : DataFrame containing wind data
    _city (str) : City name to filter
    _month (int) : Month to filter e.g. 1 for January, 2 for February, etc.

    Returns:
    DataFrame : Filtered DataFrame with average wind speed for the specified city and month
    '''
    filtered_df = _df[(_df['stnNm'].str.contains(_city)) & (_df['월'] == _month)]
    result = filtered_df.groupby('년')['ws'].mean(numeric_only=True).reset_index()
    return result
#%%
#################################################################################
ASOS = load_wind(files)
#%%
dongnam = ['부산', '울산', '대구', '진주', '창원', '김해', '양산', '구미', '영천', '포항', '경주']
#dongnam = ['하동', '고성', '칠곡', '경산']
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 

for city in dongnam:
    for month in months:
        ASOS_avg = select_data(ASOS, city, month)

        results = []
        for start_year in range(2005, 2016):  # 2015까지 포함하면 2015~2024까지 됨
            end_year = start_year + 9
            temp = ASOS_avg[(ASOS_avg['년'] >= start_year) & (ASOS_avg['년'] <= end_year)]
            mean_wind = temp['ws'].mean()
            results.append({'시작연도': start_year, '종료연도': end_year, '10년평균풍속': mean_wind})

        result_df = pd.DataFrame(results)

        plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(13,8))
        ax.plot(result_df['시작연도'], result_df['10년평균풍속'], marker='o', linestyle='-', color='r')
        ax.set_title(f'{city} {month}월 풍속 이동평균', fontsize=16)
        ax.set_ylabel('평균 풍속 (m/s)', fontsize=12)
        ax.grid(axis='y', linestyle='-', alpha=0.7) 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.set_xticks([])

        # 선형회귀, R2 계산
        x = result_df['시작연도']
        y = result_df['10년평균풍속']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_eq = f'y = {slope:.4f}x + {intercept:.4f}'

        r_squared = r_value ** 2
        r_squared_text = f'R² = {r_squared:.4f}'

        ax.plot(x, slope * x + intercept, color='k', linestyle='--', label=f'{regression_eq}\n{r_squared_text}', alpha=0.7)
        ax.legend(fontsize=12)



        df_table = pd.DataFrame({
            '연도': result_df['시작연도'].astype(str) + ' - ' + result_df['종료연도'].astype(str),
            '10년평균풍속': result_df['10년평균풍속']
        }).T

        columns = df_table.iloc[0].tolist()
        cell_text = [df_table.iloc[1].apply(lambda x: f"{x:.2f}").tolist()]

        plt.subplots_adjust(bottom=0.5)

        # 표 추가
        table = plt.table(cellText=cell_text,
                        rowLabels=[f'{month}월'],
                        rowColours=['#d0e6fa'],
                        colLabels=columns,
                        colWidths=[0.09] * len(columns),
                        colColours=['#d0e6fa'] * len(columns),
                        cellLoc='center',
                        loc='bottom')

        table.scale(1, 1.5)
        table.auto_set_font_size(False)  # 자동 글자 크기 설정 해제
        table.set_fontsize(10)

        # 표시 또는 저장
        #save_path = '/data02/dongnam/met/ASOS/data'
        save_path = '/home/hrjang2/0_code/dongnam_windavg'
        plt.savefig(f'{save_path}/{city}_{month}월_windavg.png', bbox_inches='tight', dpi=300)
        plt.show()
# %%
#################################################################################
#################################################################################
#################################################################################
#%%
AWS_path = '/data02/dongnam/met/AWS/raw/*'
files = [f for f in glob(AWS_path) if os.path.isfile(f)]
#%%
def load_wind(_files):
    '''
    Load wind data from CSV files and combine them into a single DataFrame.

    Args:
    _files (list) : List of file paths to CSV files

    Returns:
    DataFrame : Combined DataFrame containing wind data
    '''
    data_list = []
    for file in _files:
        df = pd.read_csv(file, encoding='cp949')
        df = df[['KST', 'STN', 'WS']]
        df['KST'] = pd.to_datetime(df['KST'], format = '%Y%m%d%H%M', errors='coerce')
        df['WS'] = pd.to_numeric(df['WS'], errors='coerce')
        df['STN'] = pd.to_numeric(df['STN'], errors='coerce')  
        df['년'] = df['KST'].dt.year
        df['월'] = df['KST'].dt.month
        
        data_list.append(df)
        
    combined_df = pd.concat(data_list, ignore_index=True)

    return combined_df
#%%
def select_data(_df, _month):
    '''
    Filter the wind data for a specific city and month.

    Args:
    _df (DataFrame) : DataFrame containing wind data
    _city (str) : City name to filter
    _month (int) : Month to filter e.g. 1 for January, 2 for February, etc.

    Returns:
    DataFrame : Filtered DataFrame with average wind speed for the specified city and month
    '''
    # 포항 AWS 지점번호: 804, 805, 808, 816, 830, 995
    station_regex = '804|805|808|816|830|995'
    filtered_df = _df[
        (_df['STN'].astype(str).str.contains(station_regex)) & (_df['월'] == _month)
    ]
    result = filtered_df.groupby('년')['WS'].mean(numeric_only=True).reset_index()

    return result
# %%
# --- 폰트 설정 ---
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 데이터 로드 ---
AWS = load_wind(files)
save_path = '/home/hrjang2/0_code/'
#%%
# --- 루프 시작 ---
for month in range(1, 13):
    ASOS_avg = select_data(AWS, month)

    results = []
    for start_year in range(2005, 2016):
        end_year = start_year + 9
        temp = ASOS_avg[(ASOS_avg['년'] >= start_year) & (ASOS_avg['년'] <= end_year)]
        mean_wind = temp['WS'].mean()
        results.append({'시작연도': start_year, '종료연도': end_year, '10년평균풍속': mean_wind})

    result_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(13,8))
    ax.plot(result_df['시작연도'], result_df['10년평균풍속'], marker='o', linestyle='-', color='r')
    ax.set_title(f'포항 {month}월 풍속 이동평균', fontsize=16)
    ax.set_ylabel('평균 풍속 (m/s)', fontsize=12)
    ax.grid(axis='y', linestyle='-', alpha=0.7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xticks([])

    x = result_df['시작연도']
    y = result_df['10년평균풍속']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    regression_eq = f'y = {slope:.4f}x + {intercept:.4f}'
    r_squared = r_value ** 2
    r_squared_text = f'R² = {r_squared:.4f}'

    ax.plot(x, slope * x + intercept, color='k', linestyle='--', label=f'{regression_eq}\n{r_squared_text}', alpha=0.7)
    ax.legend(fontsize=12)

    df_table = pd.DataFrame({
        '연도': result_df['시작연도'].astype(str) + ' - ' + result_df['종료연도'].astype(str),
        '10년평균풍속': result_df['10년평균풍속']
    }).T

    columns = df_table.iloc[0].tolist()
    cell_text = [df_table.iloc[1].apply(lambda x: f"{x:.2f}").tolist()]

    plt.subplots_adjust(bottom=0.5)

    table = plt.table(
        cellText=cell_text,
        rowLabels=[f'{month}월'],
        rowColours=['#d0e6fa'],
        colLabels=columns,
        colWidths=[0.09] * len(columns),
        colColours=['#d0e6fa'] * len(columns),
        cellLoc='center',
        loc='bottom'
    )

    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.savefig(f'{save_path}/포항_{month}월_windavg.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()  
# %%
# import pandas as pd
# import os

# def add_KST(file_path, save_folder):
#     filename = os.path.basename(file_path)
#     yyyymm = filename.split('_')[1][:6]

#     df = pd.read_csv(file_path, encoding='cp949')

#     nrows = len(df)
#     current_time = pd.to_datetime(yyyymm + '01' + '0000', format='%Y%m%d%H%M')

#     kst_list = []
#     previous_stn = None

#     for i in range(nrows):
#         if previous_stn is not None and df.loc[i, 'STN'] < previous_stn:
#             current_time += pd.Timedelta(hours=1)
        
#         kst_list.append(current_time.strftime('%Y%m%d%H%M'))
#         previous_stn = df.loc[i, 'STN']

#     df.insert(0, 'KST', kst_list)

#     os.makedirs(save_folder, exist_ok=True)
#     save_path = os.path.join(save_folder, filename)

#     df.to_csv(save_path, index=False, encoding='cp949')

#     return df

#%%