#%% import package
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes
import numpy as np
from scipy.stats import iqr
from AQMS_TOOL import POLLUTANT, convert_24_to_00
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False
#%% 0-1. AWS 데이터 전처리
aws_path = '/data02/dongnam/met/AWS/raw/*'
aqms_path = '/data02/dongnam/data/*.csv'

# AWS, AQMS 데이터 불러오기
years = [str(i) for i in range(2019, 2024)]
aws_files = [f for f in glob(aws_path) if any(year in f for year in years)]
aws_data = pd.concat([pd.read_csv(f) for f in aws_files], ignore_index=True)
aqms_files = [f for f in glob(aqms_path) if any(year in f for year in years)]
aqms_data = pd.concat([pd.read_csv(f) for f in aqms_files], ignore_index=True)

# AWS 데이터 전처리
'''하동읍(932) 금성면(933) AWS 데이터'''

def wind_components(ws, wd):
    wd_rad = np.deg2rad(wd)
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)
    return u, v

def uv_to_ws_wd(u, v):
    ws = np.sqrt(u**2 + v**2)
    wd = (np.rad2deg(np.arctan2(-u, -v)) + 360) % 360
    return ws, wd

aws = aws_data[aws_data['STN'].astype(str).str.contains('932|933')][['KST', 'STN', 'WD', 'WS']]
aws['KST'] = pd.to_datetime(aws['KST'], format='%Y%m%d%H%M')
aws[['U', 'V']] = aws.apply(lambda row: pd.Series(wind_components(row['WS'], row['WD'])), axis=1)
aws_stn = aws.groupby(['KST']).agg({'U': 'mean', 'V': 'mean'}).reset_index()
aws_stn[['WS', 'WD']] = aws_stn.apply(lambda row: pd.Series(uv_to_ws_wd(row['U'], row['V'])), axis=1)
aws_stn = aws_stn.drop(aws_stn.index[0])
aws_stn['KST'] = aws_stn['KST'].astype(str).apply(convert_24_to_00)
aws_result = aws_stn[['KST', 'WD', 'WS']]
#%% 0-2. aqms 데이터 전처리
def preprocess_aqms(data, pol, region_filter=None, station_filter=None):
    if region_filter:
        data = data[data['지역'].str.contains(region_filter)]

    if station_filter:
        data = data[data['측정소코드'].isin(station_filter)]

    data = data[['측정소코드', '측정일시', pol]]
    data = data.rename(columns={'측정일시': 'KST', '측정소코드': 'STN', pol: pol})

    data['KST'] = data['KST'].astype(str).apply(convert_24_to_00)
    data['KST'] = pd.to_datetime(data['KST'], format='%Y%m%d%H')

    data = data.groupby(['KST']).agg({pol: 'mean'}).reset_index()

    if pol in ['SO2', 'NO2', 'O3', 'CO']:
        data[pol] = data[pol] * 1000

    data['Year'] = data['KST'].dt.year
    data['Month'] = data['KST'].dt.month

    return data

aqms_o3 = preprocess_aqms(aqms_data, 'O3', station_filter=[238161, 238162])
aqms_no2 = preprocess_aqms(aqms_data, 'NO2', station_filter=[238161, 238162])

dongnam_aqms_o3 = preprocess_aqms(aqms_data, 'O3', region_filter='부산|대구|진주|고성|창원|김해|양산|구미|칠곡|경산|영천|포항|경주|울산')
dongnam_aqms_no2 = preprocess_aqms(aqms_data, 'NO2', region_filter='부산|대구|진주|고성|창원|김해|양산|구미|칠곡|경산|영천|포항|경주|울산')
#%% 0-3. AWS, AQMS 데이터 병합 // 동남권 데이터 처리
def preprocess_pollutant(aqms_data, aws_data, pollutant):
    aqms_data['KST'] = pd.to_datetime(aqms_data['KST'])
    aws_data['KST'] = pd.to_datetime(aws_data['KST'])

    df = pd.merge(aqms_data, aws_data, on='KST', how='inner')

    df['Year'] = df['KST'].dt.year
    df['Month'] = df['KST'].dt.month
    df['Hour'] = df['KST'].dt.hour

    df = df.dropna()
    return df

df_o3 = preprocess_pollutant(aqms_o3, aws, 'O3')
df_no2 = preprocess_pollutant(aqms_no2, aws, 'NO2')

# 데일리
df_o3['KST'] = pd.to_datetime(df_o3['KST'])
df_o3_daily = df_o3.resample('D', on='KST').mean().reset_index()
df_o3_daily = df_o3_daily.dropna()

df_no2['KST'] = pd.to_datetime(df_no2['KST'])
df_no2_daily = df_no2.resample('D', on='KST').mean().reset_index()
df_no2_daily = df_no2_daily.dropna()

#동남권
df_dongnam_o3 = dongnam_aqms_o3.copy()    
df_dongnam_o3['Year'] = df_dongnam_o3['KST'].dt.year
df_dongnam_o3['Month'] = df_dongnam_o3['KST'].dt.month
df_dongnam_o3['Hour'] = df_dongnam_o3['KST'].dt.hour
df_dongnam_o3 = df_dongnam_o3.dropna()
dongnam_o3_Q3 = df_dongnam_o3['O3'].quantile(0.75)

df_dongnam_no2 = dongnam_aqms_no2.copy()    
df_dongnam_no2['Year'] = df_dongnam_no2['KST'].dt.year
df_dongnam_no2['Month'] = df_dongnam_no2['KST'].dt.month
df_dongnam_no2['Hour'] = df_dongnam_no2['KST'].dt.hour
df_dongnam_no2 = df_dongnam_no2.dropna()
dongnam_no2_Q3 = df_dongnam_no2['NO2'].quantile(0.75)
#%% 0-4. 하동 풍향 카테고리 나누기 
def add_wd_category(df):
    conditions = [
        (df['WD'] >= 0) & (df['WD'] < 90),
        (df['WD'] >= 90) & (df['WD'] < 180),
        (df['WD'] >= 180) & (df['WD'] < 270),
        (df['WD'] >= 270) & (df['WD'] <= 360)
    ]
    choices = ['1', '2', '3', '4']
    df['category'] = np.select(conditions, choices, default=np.nan)
    return df

df_o3 = add_wd_category(df_o3)
df_no2 = add_wd_category(df_no2)
df_o3_daily = add_wd_category(df_o3_daily)
df_no2_daily = add_wd_category(df_no2_daily)
#%% 0-5. Outlier (하동, 동남권)
def remove_outliers(df, pollutant):
    Q1 = df[pollutant].quantile(0.25)
    Q2 = df[pollutant].quantile(0.50)
    Q3 = df[pollutant].quantile(0.75)
    IQR_value = iqr(df[pollutant])

    lower_bound = Q1 - 1.5 * IQR_value
    upper_bound = Q3 + 1.5 * IQR_value

    outliers = df[df[pollutant] > upper_bound]

    df_no = df.drop(outliers.index)

    return df_no, outliers, Q1, Q2, Q3, lower_bound, upper_bound

df_o3_no, outliers_o3, Q1_o3, Q2_o3, Q3_o3, lower_bound_o3, upper_bound_o3 = remove_outliers(df_o3, 'O3')
df_no2_no, outliers_no2, Q1_no2, Q2_no2, Q3_no2, lower_bound_no2, upper_bound_no2 = remove_outliers(df_no2, 'NO2')
df_o3_no_daily, outliers_o3_daily, Q1_o3_daily, Q2_o3_daily, Q3_o3_daily, lower_bound_o3_daily, upper_bound_o3_daily= remove_outliers(df_o3_daily, 'O3')
df_no2_no_daily, outliers_no2_daily, Q1_no2_daily, Q2_no2_daily, Q3_no2_daily, lower_bound_no2_daily, upper_bound_no2_daily= remove_outliers(df_no2_daily, 'NO2')

print(f'하동 Q3: {Q3_o3}, upper_bound: {upper_bound_o3}')
print(f'하동 daily O3 Q3: {int(Q3_o3_daily):02d}, upper_bound: {int(upper_bound_o3_daily):02d}')
print(f'하동 NO2 Q3: {Q3_no2}, upper_bound: {upper_bound_no2}')
print(f'하동 daily NO2 Q3: {int(Q3_no2_daily):02d}, upper_bound: {int(upper_bound_no2_daily):02d}')
print(f'동남권 O3 Q3: {int(dongnam_o3_Q3):02d}, 동남권 NO2 Q3: {int(dongnam_no2_Q3):02d}')
#%% 1-1. 하동 O3, NO2 Boxplot
def plot_boxplot_with_stats(df, pol, Q1, Q2, Q3, lower_bound, upper_bound, region_name='지역명', save_path='/data02/dongnam/output_fig/hadong0/'):

    fig, ax = plt.subplots(figsize=(4.5, 5))
    
    df.boxplot(column=pol, ax=ax)
    
    #plt.yscale('log')

    ax.text(1.1, Q3, f'Q3 = {Q3:.2f}', va='center', fontsize=10, color='blue', fontweight='bold')
    ax.text(0.9, Q2, f'Q2 (Median) = {Q2:.2f}', va='center', ha='right', fontsize=10, color='green', fontweight='bold')
    ax.text(1.1, Q1, f'Q1 = {Q1:.2f}', va='center', fontsize=10, color='blue', fontweight='bold')
    ax.text(1.1, lower_bound+0.5, f'Lower bound = {lower_bound:.2f}', va='center', fontsize=10, color='red', fontweight='bold')
    ax.text(1.1, upper_bound, f'Upper bound = {upper_bound:.2f}', va='center', fontsize=10, color='red', fontweight='bold')
    
#    for yval in [Q1, Q2, Q3, lower_bound, upper_bound]:
#        if yval > 0:  # log scale 범위 안에서만 선 긋기
#            ax.axhline(y=yval, color='gray', linestyle='--', linewidth=0.5)
    pol_name = POLLUTANT(pol).name
    ax.set_title(f'{region_name} {pol_name} Boxplot', fontsize=14)
    ax.set_ylabel(f'{pol_name} (ppb)')
    ax.set_xticklabels([f'전체 데이터({len(df)})'])
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+f'{pol}_{region_name}_boxplot.png', dpi=300)
        plt.close()
    else:
        plt.show()

plot_boxplot_with_stats(df_o3_daily, 'O3', Q1_o3_daily, Q2_o3_daily, Q3_o3_daily, lower_bound_o3_daily, upper_bound_o3_daily, region_name='하동',save_path='/data02/dongnam/output_fig/hadong0/')
plot_boxplot_with_stats(df_no2_daily, 'NO2', Q1_no2_daily, Q2_no2_daily, Q3_no2_daily, lower_bound_no2_daily, upper_bound_no2_daily, region_name='하동',save_path='/data02/dongnam/output_fig/hadong0/')

# %% 2-1. 바람장미 -> 음...........
## 오존 outlier 80.75 / daily outlier 65 // 이산화질소 outlier  21.25 / daily outlier 18
custom_bins = {
    'SO2':  [0, 4.5, 10, 15, 20, 25],
    'NO2':  [0, 5, 10, 15, 20],
    'CO':   [0, 200, 400, 600, 800],
    'O3':   [0, 20, 40, 60, 80],
    'PM10': [0, 10, 20, 30, 40, 50],
    'PM25': [0, 8, 16, 24, 32, 40],
}

def plot_windrose(df, pol, season, region_name='하동', save_dir='/data02/dongnam/output_fig/hadong0/'):
    if season == '봄':
        df = df.loc[(df['Month'] >= 3) & (df['Month'] <= 5)]
    elif season == '여름':
        df= df.loc[(df['Month'] >= 6) & (df['Month'] <= 8)]
    elif season == '가을':
        df   = df.loc[(df['Month'] >= 9) & (df['Month'] <= 11)]
    elif season == '겨울':
        df = df.loc[(df['Month'] == 12) | (df['Month'] <= 2)]
    elif season == '전체':
        df = df.copy()

    bins = custom_bins.get(pol)  

    fig = plt.figure(figsize=(6, 5))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(df['WD'], df[pol], normed=True, opening=0.8, bins=bins, alpha=0.8)

    pol_name = POLLUTANT(pol).name

    plt.title(f'{region_name} {season} 풍향별 {pol_name} 고농도 분포', fontsize=13)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=7)

    ax.set_yticks([4, 8, 12, 16, 20])
    ax.set_yticklabels(['4%', '8%', '12%', '16%', '20%'], fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{save_dir}{pol}_{season}_고농도_windrose.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_windrose(outliers_o3, 'O3', '전체')
plot_windrose(outliers_no2, 'NO2', '전체')