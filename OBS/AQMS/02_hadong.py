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
def weighted_wind_direction_mean(direction_deg, speed):
    mask = ~np.isnan(direction_deg) & ~np.isnan(speed)
    direction_deg = np.array(direction_deg)[mask]
    speed = np.array(speed)[mask]
    if len(direction_deg) == 0 or np.sum(speed) == 0:
        return np.nan

    direction_rad = np.deg2rad(direction_deg)

    x = np.sum(speed * np.cos(direction_rad)) / np.sum(speed)
    y = np.sum(speed * np.sin(direction_rad)) / np.sum(speed)

    mean_rad = np.arctan2(y, x)
    mean_deg = np.rad2deg(mean_rad)

    if mean_deg < 0:
        mean_deg += 360

    return mean_deg

def apply_weighted_mean(group):
    wd_mean = weighted_wind_direction_mean(group['WD'], group['WS'])
    ws_mean = group['WS'].mean()  # or sum(), 원하는 방식
    return pd.Series({'WD': wd_mean, 'WS': ws_mean})

aws = aws_data[aws_data['STN'].astype(str).str.contains('932|933')][['KST', 'STN', 'WD', 'WS']]
aws['KST'] = pd.to_datetime(aws['KST'], format='%Y%m%d%H%M')
aws_o3 = aws.groupby('KST').apply(apply_weighted_mean).reset_index()
aws_o3 = aws_o3.sort_values('KST').reset_index(drop=True)
aws_daily = aws.groupby(pd.Grouper(key='KST', freq='D')).apply(apply_weighted_mean).reset_index()

wd_8hr_list = []
for idx in range(len(aws_o3)):
    if idx < 7:
        wd_8hr_list.append(np.nan)
    else:
        wd_window = aws_o3['WD'].iloc[idx-7:idx+1]
        ws_window = aws_o3['WS'].iloc[idx-7:idx+1]
        wd_mean = weighted_wind_direction_mean(wd_window.values, ws_window.values)
        wd_8hr_list.append(wd_mean)

aws_o3['WD_8hr'] = wd_8hr_list
#%%
aws_933 = aws[aws.STN==932]
aws_933['Month'] = aws_933['KST'].dt.month
aws_933['Hour'] = aws_933['KST'].dt.hour

aws_933_jja = aws_933[(aws_933['Month'] >= 6) & (aws_933['Month'] <= 8)]
aws_933_jja_groupby = aws_933_jja.groupby(['Hour'])

def plot_windrose_diurnal(aws_df):

    df = aws_df.groupby(['Hour'])

    for key, group in df:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(group['WD'], group['WS'], normed=True, opening=0.8, edgecolor='white')

        ax.set_legend()
        ax.set_title(f'{key[0]:02d}', fontsize=14)

plot_windrose_diurnal(aws_933_jja)
#%% 0-2. aqms 데이터 전처리
def preprocess_aqms(data, pol, region_filter=None, station_filter=None):
    # 지역 또는 측정소 필터링
    if region_filter:
        data = data[data['지역'].str.contains(region_filter)]
    if station_filter:
        data = data[data['측정소코드'].isin(station_filter)]

    # 컬럼 정리 및 시간 처리
    data = data[['측정소코드', '측정일시', pol]].rename(columns={
        '측정일시': 'KST',
        '측정소코드': 'STN'
    })
    data['KST'] = pd.to_datetime(data['KST'].astype(str).apply(convert_24_to_00), format='%Y%m%d%H')

    # 시간별 평균, 단위 보정
    data = data.groupby('KST', as_index=False)[pol].mean()
    if pol in ['SO2', 'NO2', 'O3', 'CO']:
        data[pol] *= 1000

    # 연도 및 월 정보 추가
    data['Year'] = data['KST'].dt.year
    data['Month'] = data['KST'].dt.month

    return data

# 측정소 필터 적용
station_ids = [238161, 238162]
aqms_o3 = preprocess_aqms(aqms_data, 'O3', station_filter=station_ids)
aqms_no2 = preprocess_aqms(aqms_data, 'NO2', station_filter=station_ids)

# 일별 평균 계산
aqms_o3_daily = aqms_o3.resample('D', on='KST').mean().reset_index()
aqms_no2_daily = aqms_no2.resample('D', on='KST').mean().reset_index()

# 지역 필터 적용
dongnam_region = '부산|대구|진주|고성|창원|김해|양산|구미|칠곡|경산|영천|포항|경주|울산'
dongnam_aqms_o3 = preprocess_aqms(aqms_data, 'O3', region_filter=dongnam_region)
dongnam_aqms_no2 = preprocess_aqms(aqms_data, 'NO2', region_filter=dongnam_region)
#%% 0-3-0. AWS, AQMS 데이터 병합 // 동남권 데이터 처리
def preprocess_pollutant(aqms_data, aws_data):
    aqms_data['KST'] = pd.to_datetime(aqms_data['KST'])
    aws_data['KST'] = pd.to_datetime(aws_data['KST'])
    df = pd.merge(aqms_data, aws_data, on='KST', how='inner')
    df['Year'] = df['KST'].dt.year
    df['Month'] = df['KST'].dt.month
    df['Hour'] = df['KST'].dt.hour
    return df

def resample_daily(df, value_col):
    df['KST'] = pd.to_datetime(df['KST'])
    df_daily = df.resample('D', on='KST').mean().reset_index()
    return df_daily.dropna(), df_daily[value_col].quantile(0.75)

# 일반 O3 / NO2 데이터
df_o3 = preprocess_pollutant(aqms_o3, aws).dropna()
df_no2 = preprocess_pollutant(aqms_no2, aws).dropna()

# 일별 평균
df_no2_daily = preprocess_pollutant(aqms_no2_daily, aws_daily).dropna()
df_o3_daily = preprocess_pollutant(aqms_o3_daily, aws_daily).dropna()

# 8시간 평균 최대값 계산
aqms_o32 = preprocess_pollutant(aqms_o3, aws_o3)
aqms_o32['8hr_max'] = aqms_o32['O3'].rolling(window=8, min_periods=6).mean()
aqms_o32['KST'] = aqms_o32['KST'] - pd.Timedelta(hours=7)
aqms_o32 = aqms_o32.set_index('KST')

idx = aqms_o32.groupby(aqms_o32.index.date)['8hr_max'].idxmax().dropna().drop(index=0)
daily_max_8hr = aqms_o32.loc[idx].reset_index()
daily_max_8hr = daily_max_8hr[['KST', '8hr_max', 'WD_8hr', 'WS', 'Year', 'Month']]
daily_max_8hr.columns = ['KST', '8hr_max', 'WD', 'WS', 'Year', 'Month']

# 동남권 O3 / NO2 처리 및 Q3 계산
df_dongnam_o3, dongnam_o3_Q3 = resample_daily(dongnam_aqms_o3.copy(), 'O3')
df_dongnam_no2, dongnam_no2_Q3 = resample_daily(dongnam_aqms_no2.copy(), 'NO2')
#%% 0-4. 하동 풍향 카테고리 나누기 // MDA8 기준치 초과
def add_wd_category(df):
    """풍향을 기준으로 4개 범주로 나눔"""
    bins = [0, 90, 180, 270, 360]
    labels = ['1', '2', '3', '4']
    df['category'] = pd.cut(df['WD'], bins=bins, labels=labels, right=False, include_lowest=True)
    return df

# 풍향 카테고리 추가
for target_df in [daily_max_8hr, df_no2_daily, df_o3, df_no2]:
    add_wd_category(target_df)

# MDA8 > 60 ppb 조건으로 분리
daily_max_8hr = daily_max_8hr.reset_index(drop=True)
daily_max_8hr_only = daily_max_8hr[daily_max_8hr['8hr_max'] > 60].reset_index(drop=True)
daily_max_8hr_no = daily_max_8hr[daily_max_8hr['8hr_max'] <= 60].reset_index(drop=True)

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

df_o3, df_no2 = df_o3.dropna(), df_no2.dropna()
df_o3_no, outliers_o3, Q1_o3, Q2_o3, Q3_o3, lower_bound_o3, upper_bound_o3 = remove_outliers(df_o3, 'O3')
df_no2_no, outliers_no2, Q1_no2, Q2_no2, Q3_no2, lower_bound_no2, upper_bound_no2 = remove_outliers(df_no2, 'NO2')
df_o3_no_daily, outliers_o3_daily, Q1_o3_daily, Q2_o3_daily, Q3_o3_daily, lower_bound_o3_daily, upper_bound_o3_daily= remove_outliers(df_o3_daily, 'O3')
df_no2_no_daily, outliers_no2_daily, Q1_no2_daily, Q2_no2_daily, Q3_no2_daily, lower_bound_no2_daily, upper_bound_no2_daily= remove_outliers(df_no2_daily, 'NO2')
#%% MDA8 O3 동남권이랑 비교
'''
MDA8 O3 동남권이랑 비교
'''
df_dongnam_o3['8hr_max'] = df_dongnam_o3['O3'].rolling(window=8, min_periods=6).mean()
df_dongnam_o3['KST'] = df_dongnam_o3['KST'] - pd.Timedelta(hours=7)

mean_in = daily_max_8hr.groupby('Year')['8hr_max'].mean()        # 하동 연평균
df_dongnam_o3 = df_dongnam_o3.loc[df_dongnam_o3['KST'].dt.year < 2024]
mean_no = df_dongnam_o3.groupby('Year')['8hr_max'].mean()      # 동남권 연평균
Years = mean_in.index

plt.figure(figsize=(6, 3))
plt.plot(Years, mean_in.values, marker='o', label='하동', color='red')
plt.plot(Years, mean_no.values, marker='o', label='동남권 평균', color='black')
plt.title('MDA8 O$_3$ 연도별 평균 농도 비교', fontsize=12)
plt.xlabel('연도', fontsize=12)
plt.ylabel('MDA8 O$_3$ 농도 (ppb)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(Years)
plt.legend()
plt.tight_layout()
plt.show()

#%% 1-0. O3, NO2 통계 요약{안씀요}
def print_pollutant_summary(pol, outlier,df_in, df_no,region_name='하동'):
    if pol == 'O3':
        outlier, df_in, df_no = outliers_o3_daily, df_o3_daily, df_o3_no_daily
    elif pol == 'NO2':
        outlier, df_in, df_no  = outliers_no2_daily, df_no2_daily, df_no2_no_daily
    else:
        raise ValueError("'O3' 또는 'NO2'만 가능함요")

    h_mean_out = outlier[pol].mean()
    h_mean_in  = df_in[pol].mean()
    h_mean_no  = df_no[pol].mean()
    h_med_out = outlier[pol].median()
    h_med_in   = df_in[pol].median()
    h_med_no   = df_no[pol].median()

    pol_names = {'O3': '오존', 'NO2': '이산화질소'}
    pol_name = pol_names.get(pol, pol)

    print(f"\n📊 {region_name} {pol_name} 통계 요약 (단위:ppb)")
    print("-" * 43)
    print(f"{'구분':<10}{'Mean':>12}{'Median':>14}")
    print("-" * 43)
    print(f"{'고농도 only    ':<10}{h_mean_out:>12.3f}{h_med_out:>14.3f}")
    print(f"{'고농도 포함':<10}{h_mean_in:>12.3f}{h_med_in:>14.3f}")
    print(f"{'고농도 제외':<10}{h_mean_no:>12.3f}{h_med_no:>14.3f}")
    print("-" * 43)

print_pollutant_summary('O3', outliers_o3_daily, df_o3_daily, df_o3_no_daily)
print_pollutant_summary('NO2', outliers_no2_daily, df_no2_daily, df_no2_no_daily)
#%% 1-1. 하동 O3, NO2 Boxplot
def plot_boxplot_with_stats(df, pol, Q1, Q2, Q3, lower_bound, upper_bound, save_path='/data02/dongnam/output_fig/hadong0/'):

    fig, ax = plt.subplots(figsize=(4.5, 5))
    
    df.boxplot(column=pol, ax=ax)
    
    #plt.yscale('log')

    ax.text(1.1, Q3, f'Q3 = {Q3:.2f}', va='center', fontsize=10, color='blue', fontweight='bold')
    ax.text(0.9, Q2, f'Q2 (Median) = {Q2:.2f}', va='center', ha='right', fontsize=10, color='green', fontweight='bold')
    ax.text(1.1, Q1, f'Q1 = {Q1:.2f}', va='center', fontsize=10, color='blue', fontweight='bold')
    ax.text(1.1, lower_bound+0.5, f'Lower bound = {lower_bound:.2f}', va='center', fontsize=10, color='red', fontweight='bold')
    ax.text(1.1, upper_bound, f'Upper bound = {upper_bound:.2f}', va='center', fontsize=10, color='red', fontweight='bold')
    

    pol_name = POLLUTANT(pol).name
    ax.set_title(f'하동 {pol_name} Boxplot', fontsize=14)
    ax.set_ylabel(f'{pol_name} (ppb)')
    ax.set_xticklabels([f'전체 데이터({len(df)})'])
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig(save_path+f'{pol}_하동_boxplot.png', dpi=300)
    plt.show()

# plot_boxplot_with_stats(df_o3_daily, 'O3', Q1_o3_daily, Q2_o3_daily, Q3_o3_daily, lower_bound_o3_daily, upper_bound_o3_daily,save_path='/data02/dongnam/output_fig/hadong0/')
plot_boxplot_with_stats(df_no2_daily, 'NO2', Q1_no2_daily, Q2_no2_daily, Q3_no2_daily, lower_bound_no2_daily, upper_bound_no2_daily,save_path='/data02/dongnam/output_fig/hadong/NO2/')
#%% 1-2. O3, NO2 연도별 평균 농도
def plot_yearly_mean(df, pol, region_name='지역명',save_path=None):
    plt.figure(figsize=(4, 3))
    if pol == '8hr_max':
        h_out_mean = daily_max_8hr_only.groupby('Year')[pol].mean().drop(2024, errors='ignore')
        h_mean = daily_max_8hr.groupby('Year')[pol].mean().drop(2024, errors='ignore')
        h_no_out_mean = daily_max_8hr_no.groupby('Year')[pol].mean().drop(2024, errors='ignore')
        plt.plot(h_out_mean.index, h_out_mean.values, marker='o', label='고농도 only', color='red')
        plt.plot(h_no_out_mean.index, h_no_out_mean.values, marker='o', label='고농도 제외', color='black')
        pol_name = 'MDA8'
    else:
        pol_name = POLLUTANT(pol).name
        h_mean = df.groupby('Year')[pol].mean().drop(2024, errors='ignore')
        plt.plot(h_mean.index, h_mean.values, marker='o', label='전체 데이터', color='black')
    
    plt.title(f'연도별 {pol_name} 평균 농도', fontsize=14)
    plt.xlabel('연도', fontsize=12)
    plt.ylabel(f'{pol_name} 평균 농도 (ppb)', fontsize=12)
    plt.xticks(h_mean.index)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+f'{pol}_{region_name}_yearly.png', dpi=300)
    plt.show()

plot_yearly_mean(daily_max_8hr, '8hr_max', region_name='하동')
plot_yearly_mean(df_no2_daily, 'NO2', region_name='하동')
#%% 1-3. O3, NO2 월별 평균 농도
def plot_yearmonthly(df, outlier, no_outlier, pol, save_path=None):
    # 연-월별 평균
    if pol == '8hr_max':
        pol_name = 'MDA8'
    else:
        pol_name = POLLUTANT(pol).name
    h_mean = df.groupby(['Year', 'Month'])[pol].mean().reset_index()
    h_out_mean = outlier.groupby(['Year', 'Month'])[pol].mean().reset_index()
    h_no_out_mean = no_outlier.groupby(['Year', 'Month'])[pol].mean().reset_index()
        
    for df in [h_mean, h_out_mean, h_no_out_mean]:
        df['YearMonth'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + df['Month'].astype(int).astype(str).str.zfill(2))
        df.set_index('YearMonth', inplace=True)

    plt.figure(figsize=(8, 3))

    plt.plot(h_out_mean.index, h_out_mean[pol], marker='o', label='고농도 only', color='red')
    plt.plot(h_no_out_mean.index, h_no_out_mean[pol], marker='o', label='고농도 제외', color='black')
    plt.title(f'{pol_name} 평균 농도', fontsize=14)
    plt.ylabel(f'{pol_name} 농도 (ppb)', fontsize=12)
    plt.xticks(h_mean.index[::max(len(h_mean)//15,1)],
               h_mean.index[::max(len(h_mean)//15,1)].strftime('%Y%m'), rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path+f'{pol}_하동_yearmonth.png', dpi=300)
    plt.show()

plot_yearmonthly(daily_max_8hr, daily_max_8hr_only, daily_max_8hr_no, '8hr_max')
plot_yearmonthly(df_o3_daily, outliers_o3_daily, df_o3_no_daily, 'O3')
plot_yearmonthly(df_no2_daily, outliers_no2_daily, df_no2_no_daily, 'NO2')
#%% 1-4. O3, NO2 히스토그램
def plot_histogram(df_in, df_no, pol, region_name='하동', save_path=None):
    plt.figure(figsize=(5, 3))
    if pol == '8hr_max':
        pol_name = 'MDA8'
    else:
        pol_name = POLLUTANT(pol).name
    bins = np.histogram_bin_edges(pd.concat([df_in[pol], df_no[pol]]).dropna(), bins=30)

    # 고농도 포함
    sns.histplot(data=df_in, x=pol, bins=bins, element='step', stat='count',
                 common_norm=False, color='gray', label='고농도 포함')

    # 고농도 제외
    sns.histplot(data=df_no, x=pol, bins=bins, element='step', stat='count',
                 common_norm=False, color='skyblue', label='고농도 제외')


    plt.title(f'{region_name} {pol_name} 분포', fontsize=14)
    plt.xlabel(f'{pol_name} 농도 (ppb)', fontsize=12)
    plt.ylabel('frequency (log)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+f'{pol}_{region_name}_histogram_log.png', dpi=300)
        plt.show()
    else:
        plt.show()

plot_histogram(daily_max_8hr, daily_max_8hr_no, '8hr_max', region_name='하동', save_path='/data02/dongnam/output_fig/hadong/O3_MDA8/')
# plot_histogram(df_o3_daily, df_o3_no_daily, 'O3')
plot_histogram(df_no2_daily, df_no2_no_daily, 'NO2', save_path='/data02/dongnam/output_fig/hadong/NO2/')    
#%% 1-5. O3, NO2 증가율. 차이?
def plot_yearly_trend(df_in, df_no, pol, save_path=None):
    mean_in = df_in.groupby('Year')[pol].mean()
    mean_no = df_no.groupby('Year')[pol].mean()

    diff_pct = ((mean_in - mean_no) / mean_in * 100).round(2)
    diff_pct.index = diff_pct.index.astype(int)
    if pol == '8hr_max':
        pol_name = 'MDA8'
    else:
        pol_name = POLLUTANT(pol).name

    # 증가율 plot
    ax = diff_pct.plot(kind='bar', color='skyblue', figsize=(5, 3))
    plt.title(f'{pol_name} 농도 증가율 (%)', fontsize=12)
    plt.ylabel('Difference (%)')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')

    for i, v in enumerate(diff_pct.values):
        ax.text(i, v / 2 if v != 0 else 0, f'{v:.1f}%', ha='center', va='center', color='black', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}{pol}_growthrate.png', dpi=300)
        plt.close()
    else:
        plt.show()

plot_yearly_trend(daily_max_8hr, daily_max_8hr_no, '8hr_max', save_path='/data02/dongnam/output_fig/hadong/O3_MDA8/')
# plot_yearly_trend(df_o3_daily, df_o3_no_daily, 'O3', save_path='/data02/dongnam/output_fig/hadong0/')
plot_yearly_trend(df_no2_daily, df_no2_no_daily, 'NO2', save_path='/data02/dongnam/output_fig/hadong/NO2/')
# %% 2-1. 바람장미 -> 음...........
## 오존 daily outlier 65.20549242424244 // 이산화질소 daily outlier 18.85606060606061
def plot_windrose(df, pol, season,region_name='하동', outlier=False):
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

    fig = plt.figure(figsize=(6, 5))
    ax = WindroseAxes.from_ax(fig=fig)

    if pol == 'O3':
        bins = [65, 68, 70, 72, 74]
        pol_name = POLLUTANT(pol).name
    elif pol == 'NO2':
        bins = [18, 20, 22, 24, 26]
        pol_name = POLLUTANT(pol).name
    elif pol == '8hr_max':
        bins = (0, 20, 40, 60, 80)
        pol_name = 'MDA8'
    else:
        raise ValueError("Invalid pollutant. Choose 'O3', 'NO2', or '8hr_max'.")
    
    if outlier==True:
        if pol == 'O3':
            bins = [65, 68, 70, 72, 74]
            pol_name = POLLUTANT(pol).name
        elif pol == 'NO2':
            bins = [18, 20, 22, 24, 26]
            pol_name = POLLUTANT(pol).name
        elif pol == '8hr_max':
            bins = (60, 70, 80, 90, 100)
            pol_name = 'MDA8'
        plt.title(f'{region_name} {season} 풍향별 {pol_name} 고농도 분포({len(df)})', fontsize=13)
        ax.bar(df['WD'], df[pol], bins=bins, normed=True, opening=0.8, edgecolor='black', linewidth=0.5,
            cmap=plt.cm.viridis, alpha=0.7)
        ax.set_yticks([5, 10, 15, 20, 25])
        ax.set_yticklabels(['5%','10%','15%','20%','25%'], fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=7)
        plt.tight_layout()
        plt.savefig(f'/data02/dongnam/output_fig/hadong/{pol}_{season}_고농도_windrose.png', dpi=300)
        plt.show()
    if outlier==False:
        if pol == 'O3':
            bins = [65, 68, 70, 72, 74]
            pol_name = POLLUTANT(pol).name
        elif pol == 'NO2':
            bins = [0, 5, 10, 15, 20]
            pol_name = POLLUTANT(pol).name
        elif pol == '8hr_max':
            bins = (0, 20, 40, 60, 80)
            pol_name = 'MDA8'
        plt.title(f'{region_name} {season} 풍향별 {pol_name} 농도 분포({len(df)})', fontsize=13)
        ax.bar(df['WD'], df[pol], bins=bins, normed=True, opening=0.8, edgecolor='black', linewidth=0.5,
            cmap=plt.cm.viridis, alpha=0.7)
        ax.set_yticks([5, 10, 15, 20, 25])
        ax.set_yticklabels(['5%','10%','15%','20%','25%'], fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=7)
        plt.tight_layout()
        plt.savefig(f'/data02/dongnam/output_fig/hadong/{pol}_{season}_windrose.png', dpi=300)
        plt.show()

plot_windrose(daily_max_8hr, '8hr_max', '겨울', outlier=False)
plot_windrose(daily_max_8hr_only, '8hr_max', '겨울', outlier=True)
plot_windrose(outliers_o3_daily, 'O3', '겨울', outlier=True)
plot_windrose(df_no2_daily, 'NO2', '겨울', outlier=False)
plot_windrose(outliers_no2_daily, 'NO2', '겨울', outlier=True)
#%% 2-2. 풍향 카테고리별 갯수
def category_count(df, pol):
    if pol == '8hr_max':
        thresholds = [0, 20, 40, 60, 80]  # 농도 구간
        labels = ['0-20', '20-40', '40-60', '60-80', '80+']  # 레이블
    elif pol == 'NO2':
        thresholds = [0, 5, 10, 15, 20]  # 농도 구간
        labels = ['0-5', '5-10', '10-15', '15-20', '20+']  # 레이블

    bins_df = pd.DataFrame(0, index=['1', '2', '3', '4'], columns=labels)
    bins_df.index.name = 'category'
    bins_df.columns.name = 'label'

    for cat in ['1', '2', '3', '4']:
        sub = df[df['category'] == cat]
        for i in range(len(thresholds)):
            lower = thresholds[i]
            upper = thresholds[i+1] if i+1 < len(thresholds) else np.inf
            count = sub[(sub[pol] >= lower) & (sub[pol] < upper)].shape[0]
            bins_df.loc[cat, labels[i]] = count
    return print(bins_df)
print('-----------------------------------------')
category_count(daily_max_8hr, '8hr_max')
print('-----------------------------------------')
category_count(df_no2_daily, 'NO2')
print('-----------------------------------------')
#%% 2-3. 계절별
def get_season(month):
    if month in [3,4,5]:
        return '봄'
    elif month in [6,7,8]:
        return '여름'
    elif month in [9,10,11]:
        return '가을'
    else:
        return '겨울'
df_no2_daily['Season'] = df_no2_daily['Month'].apply(get_season)
daily_max_8hr['Season'] = daily_max_8hr['Month'].apply(get_season)
daily_max_8hr_only['Season'] = daily_max_8hr_only['Month'].apply(get_season)


def season_count(df, pol):
    if pol == '8hr_max':
        thresholds = [0, 20, 40, 60, 80]  # 농도 구간
        labels = ['0-20', '20-40', '40-60', '60-80', '80+']  # 레이블
    elif pol == 'NO2':
        thresholds = [0, 5, 10, 15, 20]  # 농도 구간
        labels = ['0-5', '5-10', '10-15', '15-20', '20+']  # 레이블

    bins_df = pd.DataFrame(0, index=['봄', '여름', '가을', '겨울'], columns=labels)
    bins_df.index.name = 'season'
    bins_df.columns.name = 'label'

    for season in ['봄', '여름', '가을', '겨울']:
        sub = df[df['Season'] == season]
        for i in range(len(thresholds)):
            lower = thresholds[i]
            upper = thresholds[i+1] if i+1 < len(thresholds) else np.inf
            count = sub[(sub[pol] >= lower) & (sub[pol] < upper)].shape[0]
            bins_df.loc[season, labels[i]] = count
    return bins_df

print(season_count(daily_max_8hr, '8hr_max'))
print('-----------------------------------------')
print(season_count(df_no2_daily, 'NO2'))
#%% 계절별_고농도_관측_횟수
colors = {'봄': '#ff9999', '여름': '#ffcc80', '가을': '#a3d9a5', '겨울': '#99ccff'}
def plot_yearly_seasonal_count(df, pol):
    if pol == '8hr_max':
        pol_name = 'MDA8'
    else:
        pol_name = pol

    # 계절 순서 지정
    season_order = ['봄', '여름', '가을', '겨울']
    df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)

    df_count = df.groupby(['Year', 'Season']).size().reset_index(name='Count')

    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(data=df_count, x='Year', y='Count', hue='Season', hue_order=season_order, palette=colors, ax=ax)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.text(p.get_x() + p.get_width() / 2, height/2, int(height), ha='center', va='center', fontsize=9)

    ax.set_title(f'{pol_name} 연도별 계절별 고농도 관측 횟수')
    ax.set_xlabel('연도')
    ax.set_ylabel('관측 횟수')
    ax.legend(fontsize=8)
    plt.tight_layout()
    # plt.savefig(f'/data02/dongnam/output_fig/hadong/O3_MDA8/{pol}_계절별_고농도_관측_횟수.png', dpi=300)
    plt.show()

plot_yearly_seasonal_count(daily_max_8hr_only, '8hr_max')
#%% 요일별
weekday_order = ['월', '화', '수', '목', '금', '토', '일']
weekday_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}

daily_max_8hr_only['요일'] = daily_max_8hr_only['KST'].dt.weekday.map(weekday_map)
df_no2_daily['요일'] = df_no2_daily['KST'].dt.weekday.map(weekday_map)
# =============================
# ✅ 연도별 요일별 MDA8 발생 횟수
# =============================
count_by_weekday = daily_max_8hr_only.groupby(['요일', 'Year']).size().unstack(fill_value=0)
count_by_weekday = count_by_weekday.loc[weekday_order]
count_by_year = count_by_weekday.T  # index: 연도, columns: 요일

ax1 = count_by_year.plot(kind='bar', figsize=(6, 3))
plt.title('연도별 요일별 MDA8 발생 횟수')
plt.xlabel('연도')
plt.ylabel('발생 횟수')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='-', alpha=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 0.8), loc='upper left', borderaxespad=0)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/연도별_요일별_MDA8_횟수.png', dpi=300)
plt.show()

# =============================
# ✅ 연도별 요일별 MDA8 평균 농도
# =============================
mean_by_weekday = daily_max_8hr_only.groupby(['요일', 'Year'])['8hr_max'].mean().unstack(fill_value=0)
mean_by_weekday = mean_by_weekday.loc[weekday_order]
mean_by_year = mean_by_weekday.T  # index: 연도, columns: 요일

ax2 = mean_by_year.plot(kind='bar', figsize=(6, 3))
plt.title('연도별 요일별 MDA8 평균 농도')
plt.xlabel('연도')
plt.ylabel('MDA8 평균 농도 (ppb)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='-', alpha=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 0.8), loc='upper left', borderaxespad=0)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/연도별_요일별_MDA8_평균농도.png', dpi=300)
plt.show()


# =============================
# ✅ 연도별 요일별 NO2 평균 농도
# =============================

mean_by_weekday = df_no2_daily.groupby(['요일', 'Year'])['NO2'].mean().unstack(fill_value=0)
mean_by_weekday = mean_by_weekday.loc[weekday_order]
mean_by_year = mean_by_weekday.T  # index: 연도, columns: 요일

ax2 = mean_by_year.plot(kind='bar', figsize=(6, 3))
plt.title('연도별 요일별 NO2 평균 농도')
plt.xlabel('연도')
plt.ylabel('NO2 평균 농도 (ppb)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='-', alpha=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 0.8), loc='upper left', borderaxespad=0)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/NO2/연도별_요일별_NO2_평균농도.png', dpi=300)
plt.show()

#%%
# =============================
# ✅ 계졀별 MDA8 평균 농도 Boxplot
# =============================
plt.figure(figsize=(6,3))
sns.boxplot(data=daily_max_8hr_only, x='Year', y='8hr_max', hue='Season', palette=colors)
plt.title('MDA8 고농도 분포')
plt.xlabel('연도')
plt.ylabel('MDA8 농도 (ppb)')
plt.legend(fontsize=6)
plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/계절별_boxplot.png', dpi=300)
plt.show()

plt.figure(figsize=(6,3))
sns.boxplot(data=df_no2_daily, x='Year', y='NO2', hue='Season', palette=colors)
plt.title('NO2 고농도 분포')
plt.xlabel('연도')
plt.ylabel('NO2 농도 (ppb)')
plt.legend(fontsize=6)
plt.savefig('/data02/dongnam/output_fig/hadong/NO2/요일별_boxplot.png', dpi=300)
plt.show()

# =============================
# ✅ 계절별 MDA8 평균 농도 수준
# =============================
df_season_avg = daily_max_8hr_only.groupby(['Year', 'Season'])['8hr_max'].mean().reset_index()
plt.figure(figsize=(6,3))
sns.lineplot(data=df_season_avg, x='Year', y='8hr_max', hue='Season', marker='o', palette=colors)

plt.title('연도별 계절별 MDA8 고농도 수준')
plt.xlabel('연도')
plt.ylabel('MDA8 농도 (ppb)')
plt.xticks([2019, 2020, 2021, 2022, 2023])
plt.grid(alpha=0.7)
plt.legend(fontsize=8)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/계절별_고농도_수준.png', dpi=300)
plt.show()

df_season_avg_no2 = df_no2_daily.groupby(['Year', 'Season'])['NO2'].mean().reset_index()
plt.figure(figsize=(6,3))
sns.lineplot(data=df_season_avg_no2, x='Year', y='NO2', hue='Season', marker='o', palette=colors)

plt.title('연도별 계절별 NO2 농도 수준')
plt.xlabel('연도')
plt.ylabel('NO2 농도 (ppb)')
plt.xticks([2019, 2020, 2021, 2022, 2023])
plt.grid(alpha=0.7)
plt.legend(fontsize=8)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/NO2/계절별_고농도_수준.png', dpi=300)
plt.show()


# =============================
# ✅ 계절별 MDA8 초과율
# =============================

daily_max_8hr['초과여부'] = daily_max_8hr['8hr_max'] > 60
total_by_season = daily_max_8hr.groupby(['Year', 'Season']).size()
exceed_by_season = daily_max_8hr[daily_max_8hr['초과여부']].groupby(['Year', 'Season']).size()
season_exceed_ratio = (exceed_by_season / total_by_season * 100).round(2)

season_exceed_df = season_exceed_ratio.reset_index()
season_exceed_df.columns = ['Year', 'Season', '초과율(%)']

plt.figure(figsize=(5, 3))
sns.lineplot(data=season_exceed_df, x='Year', y='초과율(%)', hue='Season', marker='o', 
             hue_order=['봄', '여름', '가을', '겨울'], palette=colors)
plt.title('연도별 계절별 MDA8 초과율 추이')
plt.xlabel('연도')
plt.ylabel('초과율 (%)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.xticks([2019, 2020, 2021, 2022, 2023])
plt.legend(fontsize=8)
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/계절별_MDA8_초과율.png', dpi=300)
plt.show()

# %%연도별_전체_MDA8_낮밤_비율
daily_max_8hr['Time_Label'] = daily_max_8hr['KST'].dt.hour.apply(lambda h: '낮' if 7 <= h < 19 else '밤')
daily_max_8hr['Year'] = daily_max_8hr['KST'].dt.year.astype(int)

count_by_year = daily_max_8hr.groupby(['Year', 'Time_Label']).size().unstack(fill_value=0)

percent_by_year = count_by_year.div(count_by_year.sum(axis=1), axis=0) * 100

ax = percent_by_year.plot(kind='bar', stacked=True, figsize=(6, 3), color=['skyblue', 'orange'])

plt.title('연도별 전체 MDA8 낮/밤 발생 비율 (%)')
plt.xlabel('연도')
plt.ylabel('비율 (%)')
plt.xticks(rotation=0)

for i, (year, row) in enumerate(percent_by_year.iterrows()):
    낮 = row.get('낮', 0)
    밤 = row.get('밤', 0)
    ax.text(i, 낮 / 2, f'{낮:.1f}%', ha='center', va='center', fontsize=10, color='black')
    ax.text(i, 낮 + 밤 / 2, f'{밤:.1f}%', ha='center', va='center', fontsize=10, color='black')

plt.grid(axis='y', linestyle='-', alpha=0.3)
plt.legend().set_visible(False)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/연도별_전체_MDA8_낮밤_비율.png', dpi=300)
plt.show()


#%% 시간대별 패턴
daily_max_8hr_no['Hour'] = daily_max_8hr_no['KST'].dt.hour.astype(int)
daily_max_8hr_only['Hour'] = daily_max_8hr_only['KST'].dt.hour.astype(int)

hourly_mean = daily_max_8hr_only.groupby('Hour')['8hr_max'].mean().reset_index()
hourly_mean2 = daily_max_8hr_no.groupby(['Hour'])['8hr_max'].mean().reset_index()
hourly_year_mean = daily_max_8hr_only.groupby(['Hour', 'Year'])['8hr_max'].mean().unstack(fill_value=0)
hourly_year_mean2 = daily_max_8hr.groupby(['Hour', 'Year'])['8hr_max'].mean().unstack(fill_value=0)

plt.figure(figsize=(7, 3))
sns.lineplot(data=hourly_mean, x='Hour', y='8hr_max', marker='o', color='red', label='고농도 only')
sns.lineplot(data=hourly_mean2, x='Hour', y='8hr_max', marker='o', color='black', label='고농도 제외')
plt.title('시간대별 MDA8 O$_3$ 농도')
plt.xlabel('Hour')
plt.xticks(range(0, 24, 1))
plt.ylabel('MDA8 O$_3$ (ppb)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/시간대별_고농도_MDA8농도.png', dpi=300)
plt.show()

'''
MDA8 발생 비율 heatmap
'''
# 시간대별 MDA8 발생 횟수 -> 비율로 바꿔놓움
hourly_counts = daily_max_8hr_only.groupby(['Hour', 'Year']).size().unstack(fill_value=0)
hourly_percent = hourly_counts.div(hourly_counts.sum(axis=0), axis=1) * 100

sns.heatmap(hourly_percent, annot=True, fmt=".1f", cmap='hot_r')
plt.title('시간대별 고농도 MDA8 발생 비율')
plt.xlabel('연도')
plt.ylabel('시간대')
plt.yticks(rotation=0)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/시간대별_고농도_MDA8_발생비율.png', dpi=300)
plt.show()

'''
시간대별 MDA8 평균 농도
'''
daily_max_8hr_only['Hour'] = daily_max_8hr_only['KST'].dt.hour.astype(int)
hourly_mean = daily_max_8hr_only.groupby(['Hour', 'Year'])['8hr_max'].mean().unstack(fill_value=0)

plt.figure(figsize=(7, 5))
sns.heatmap(hourly_mean, annot=True, fmt=".1f", vmin=60,cmap='hot_r', cbar_kws={'label': 'MDA8 농도 (ppb)'})
plt.title('시간대별 고농도 MDA8 평균 농도')
plt.xlabel('연도')
plt.ylabel('시간대')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/시간대별_고농도_MDA8_평균농도.png', dpi=300)
plt.show()
# %% 연-월별 초과일 수
month_exceed_count = daily_max_8hr_only.groupby(['Year', 'Month']).size().reset_index(name='초과일수')

month_exceed_pivot = month_exceed_count.pivot(index='Month', columns='Year', values='초과일수').fillna(0)

yearly_total = month_exceed_pivot.sum(axis=0).astype(int)

month_exceed_pivot.columns = [f"{year}({yearly_total[year]})" for year in month_exceed_pivot.columns]

plt.figure(figsize=(6.5, 4))
sns.heatmap(month_exceed_pivot, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': '초과일 수'})
plt.title('연도-월별 MDA8 초과일 수')
plt.xlabel('연도 (총 초과일 수)')
plt.ylabel('월')
plt.yticks(rotation=0)
plt.tight_layout()
# plt.savefig('/data02/dongnam/output_fig/hadong/O3_MDA8/연도별_월별_초과일수.png', dpi=300)
plt.show()

# %% o3, no2 hourly data 사용
'''
NOₓ-limited = VOC는 충분, NOₓ이 부족 → NOₓ 늘리면 O₃ 생성 증가
VOC-limited	= NOₓ는 많고 VOC가 부족 → NOₓ 줄이면 O₃ 생성 증가 가능
Transition	=둘 다 적절한 수준 → 민감도 불명확
'''
daytime = pd.merge(df_o3, df_no2, on='KST')
daytime['Hour'] = daytime['Hour_x'] 
daytime = daytime.drop(columns=['Hour_x', 'Hour_y'])
daytime['Month'] = daytime['Month_x']
daytime = daytime.drop(columns=['Month_x', 'Month_y'])
daytime = daytime[(daytime['Hour'] >= 8) & (daytime['Hour'] <= 17)]

# 산점도 + 회귀선
plt.figure(figsize=(6, 5))
sns.regplot(data=daytime, x='NO2', y='O3', scatter_kws={'alpha':0.6}, line_kws={'color':'black'})
plt.title('낮 시간대 (11~17시) O$_3$ vs NO$_2$')
plt.xlabel('NO$_2$ 농도 (ppb)')
plt.ylabel('O$_3$ 농도 (ppb)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

corr = daytime[['O3', 'NO2']].corr().iloc[0,1]
print(f"O3 vs NO2 상관계수 (11-17시): {corr:.2f}")

for month in sorted(daytime['Month'].unique()):
    monthly = daytime[daytime['Month'] == month]
    corr = monthly[['O3', 'NO2']].corr().iloc[0,1]
    if corr > 0:
        print(f"{month}월 상관계수: {corr:.2f}  NOₓ-limited")
    elif corr < 0:
        print(f"{month}월 상관계수: {corr:.2f}  VOC-limited")
    else:
        print(f"{month}월 상관계수: {corr:.2f}  Transition")
# %% 일최대 O₃ 계산
daily_max_o3 = df_o3.groupby(df_o3['KST'].dt.date)['O3'].max().reset_index()
daily_max_o3.columns = ['KST', 'O3_max']

exceed_days = daily_max_o3[daily_max_o3['O3_max'] > 100]['KST']

df_high = df_no2[df_no2['KST'].dt.date.isin(exceed_days)]
summary = df_high.groupby(df_high['KST'].dt.date).agg({
    'NO2': 'mean',
    'WS': 'mean'
}).reset_index()

summary.columns = ['KST', 'NO2_mean', 'WS_mean']
summary = summary.merge(daily_max_o3, on='KST')
summary = summary.round(2)
# %% hourly trend
# df_no2['Hour'] = df_no2['KST'].dt.hour.astype(int)
numeric_cols = df_no2.select_dtypes(include='number').columns
hourly_mean_no2 = df_no2.groupby('Hour')[numeric_cols].mean().reset_index()
hourly_counts = df_no2.groupby('Hour').size().reset_index(name='Count')

plt.figure(figsize=(6, 3))
sns.lineplot(data=hourly_mean_no2, x='Hour', y='NO2', marker='o', color='black')
plt.title('NO$_2$ 농도')
plt.xlabel('시간')
plt.xticks(range(0, 24, 1)) 
plt.ylabel('NO$_2$ 농도 (ppb)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# %%
outliers_no2_daily['Season'] = outliers_no2_daily['Month'].apply(get_season)
seasonal_counts = outliers_no2_daily.groupby(['Season', 'category']).size().reset_index(name='Count')

seasons = ['봄', '여름', '가을', '겨울']
seasonal_counts = seasonal_counts[seasonal_counts['category'] != 'nan']
categories = sorted(seasonal_counts['category'].unique())

bar_width = 0.6
x = np.arange(len(seasons)) 
bottom = np.zeros(len(seasons))

colors = ['#ff9999', '#ffcc80', '#a3d9a5', '#99ccff']

fig, ax = plt.subplots(figsize=(7,4))

for i, cat in enumerate(categories):
    cat_data = seasonal_counts[seasonal_counts['category'] == cat].set_index('Season').reindex(seasons).fillna(0)

    counts = cat_data['Count'].values

    bars = ax.bar(x, counts, bar_width, bottom=bottom, label=str(cat), color=colors[i])

    for j, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bottom[j] + height/2,
                    f'{int(height)}', ha='center', va='center', fontsize=10)

    bottom += counts

season_counts_dict = seasonal_counts.groupby('Season')['Count'].sum().reindex(seasons).fillna(0).astype(int).to_dict()
xtick_labels = [f'{season}({season_counts_dict[season]})' for season in seasons]

ax.set_xticks(x)
ax.set_xticklabels(xtick_labels, fontsize=12)
ax.set_ylabel('Count')
ax.set_title('하동 계절별 고농도 NO$_2$ 카테고리 분포')
ax.legend(title='category')
plt.tight_layout()
plt.savefig('/data02/dongnam/output_fig/hadong/NO2/계절별_고농도_NO2_카테고리분포.png', dpi=300)
plt.show()
# %%
all_Months = pd.Index(range(1, 13))
outlier = outliers_no2_daily.groupby('Month')['NO2'].mean().reindex(all_Months).fillna(np.nan)
mean_in = df_no2_daily.groupby('Month')['NO2'].mean()
mean_no = df_no2_no_daily.groupby('Month')['NO2'].mean()

Months = mean_in.index.astype(int).tolist()
outlier = outlier.values
with_outlier = mean_in.values
without_outlier = mean_no.values

plt.figure(figsize=(6, 3))
plt.plot(Months, outlier, marker='o', label='고농도만', color='red')
plt.plot(Months, with_outlier, marker='o', label='전체 평균', color='black')
plt.plot(Months, without_outlier, marker='o', label='고농도 제외', color='blue')
plt.title('NO$_2$ 월별 평균 농도 비교', fontsize=12)
plt.xlabel('월', fontsize=12)
plt.ylabel('NO$_2$ 평균 농도 (ppb)', fontsize=12)
plt.xticks(Months)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
# plt.savefig('/home/hrjang2/0_code/hadong/하동_Monthly_NO2_비교.png', dpi=300)
plt.show()
# %%
