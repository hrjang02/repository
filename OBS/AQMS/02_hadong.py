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
df_dongnam_o3['KST'] = pd.to_datetime(df_dongnam_o3['KST'])
df_dongnam_o3 = df_dongnam_o3.resample('D', on='KST').mean().reset_index()
df_dongnam_o3 = df_dongnam_o3.dropna()
dongnam_o3_Q3 = df_dongnam_o3['O3'].quantile(0.75)

df_dongnam_no2 = dongnam_aqms_no2.copy()    
df_dongnam_no2['KST'] = pd.to_datetime(df_dongnam_no2['KST'])
df_dongnam_no2 = df_dongnam_no2.resample('D', on='KST').mean().reset_index()
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
#%% 1-0. O3, NO2 통계 요약
def print_pollutant_summary(pol, outlier,df_in, df_no,region_name='하동'):
    if pol == 'O3':
        outlier = outliers_o3_daily.copy()
        df_in = df_o3_daily.copy()
        df_no = df_o3_no_daily.copy()
    elif pol == 'NO2':
        outlier = outliers_no2_daily.copy()
        df_in = df_no2_daily.copy()
        df_no = df_no2_no_daily.copy()
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

print(f'동남권 O3 Q3: {int(dongnam_o3_Q3):02d}, 동남권 NO2 Q3: {int(dongnam_no2_Q3):02d}')
print_pollutant_summary('O3', outliers_o3_daily, df_o3_daily, df_o3_no_daily)
print_pollutant_summary('NO2', outliers_no2_daily, df_no2_daily, df_no2_no_daily)
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
#%% 1-2. O3, NO2 연도별 평균 농도
def plot_yearly_mean(df, outlier, no_outlier, pol, region_name='지역명',save_path=None):
    h_mean = df.groupby('Year')[pol].mean().drop(2024, errors='ignore')
    h_out_mean = outlier.groupby('Year')[pol].mean().drop(2024, errors='ignore')
    h_no_out_mean = no_outlier.groupby('Year')[pol].mean().drop(2024, errors='ignore')

    plt.figure(figsize=(4, 3))
    plt.plot(h_mean.index, h_mean.values, marker='o', label='전체 데이터', color='black')
    plt.plot(h_out_mean.index, h_out_mean.values, marker='o', label='고농도 only', color='red')
    plt.plot(h_no_out_mean.index, h_no_out_mean.values, marker='o', label='고농도 제외', color='blue')

    pol_name = POLLUTANT(pol).name
    plt.title(f'연도별 {pol_name} 평균 농도', fontsize=14)
    plt.xlabel('연도', fontsize=12)
    plt.ylabel(f'{pol_name} 평균 농도 (ppb)', fontsize=12)
    plt.xticks(h_mean.index)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+f'{pol}_{region_name}_yearly.png', dpi=300)
        plt.close()
    else:
        plt.show()

plot_yearly_mean(df_o3_daily, outliers_o3_daily, df_o3_no_daily, 'O3', region_name='하동')
plot_yearly_mean(df_no2_daily, outliers_no2_daily, df_no2_no_daily, 'NO2', region_name='하동')
#%% 1-3. O3, NO2 월별 평균 농도
def plot_yearmonthly(df, outlier, no_outlier, pol, region_name='지역명', save_path=None):
    # 연-월별 평균
    h_mean = df.groupby(['Year', 'Month'])[pol].mean().reset_index()
    h_out_mean = outlier.groupby(['Year', 'Month'])[pol].mean().reset_index()
    h_no_out_mean = no_outlier.groupby(['Year', 'Month'])[pol].mean().reset_index()


    # 연-월 datetime index 생성
    for df in [h_mean, h_out_mean, h_no_out_mean]:
        df['YearMonth'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + df['Month'].astype(int).astype(str).str.zfill(2))
        df.set_index('YearMonth', inplace=True)

    plt.figure(figsize=(8, 3))

    plt.plot(h_out_mean.index, h_out_mean[pol], marker='o', label='고농도 only', color='red')
    plt.plot(h_mean.index, h_mean[pol], marker='o', label='전체 데이터', color='black')
    plt.plot(h_no_out_mean.index, h_no_out_mean[pol], marker='o', label='고농도 제외', color='blue', alpha=0.7)
    
    pol_name = POLLUTANT(pol).name

    plt.title(f'연-월별 {pol_name} 평균 농도', fontsize=14)
    plt.ylabel(f'{pol_name} 농도 (ppb)', fontsize=12)
    plt.xticks(h_mean.index[::max(len(h_mean)//15,1)],
               h_mean.index[::max(len(h_mean)//15,1)].strftime('%Y%m'), rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+f'{pol}_{region_name}_yearmonth.png', dpi=300)
        plt.close()
    else:
        plt.show()


plot_yearmonthly(df_o3_daily, outliers_o3_daily, df_o3_no_daily, 'O3', region_name='하동')
plot_yearmonthly(df_no2_daily, outliers_no2_daily, df_no2_no_daily, 'NO2', region_name='하동')
#%% 1-4. O3, NO2 히스토그램
def plot_histogram(df_in, df_no, pol, region_name='하동', save_path=None):
    plt.figure(figsize=(5, 3))
    bins = np.histogram_bin_edges(pd.concat([df_in[pol], df_no[pol]]).dropna(), bins=30)

    # 고농도 포함
    sns.histplot(data=df_in, x=pol, bins=bins, element='step', stat='count',
                 common_norm=False, color='gray', label='고농도 포함')

    # 고농도 제외
    sns.histplot(data=df_no, x=pol, bins=bins, element='step', stat='count',
                 common_norm=False, color='skyblue', label='고농도 제외')

    pol_names = {'SO2': 'SO$_2$', 'O3': 'O$_3$', 'NO2': 'NO$_2$'}
    pol_label = pol_names.get(pol, pol)

    plt.title(f'{region_name} {pol_label} 분포', fontsize=14)
    plt.xlabel(f'{pol_label} 농도 (ppb)', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    #plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+f'{pol}_{region_name}_histogram_NOTlog.png', dpi=300)
        plt.close()
    else:
        plt.show()

plot_histogram(df_o3_daily, df_o3_no_daily, 'O3')
plot_histogram(df_no2_daily, df_no2_no_daily, 'NO2')
#%% 1-5. O3, NO2 증가율. 차이?
def plot_yearly_trend(df_in, df_no, pol, save_path=None):
    mean_in = df_in.groupby('Year')[pol].mean()
    mean_no = df_no.groupby('Year')[pol].mean()

    diff_pct = ((mean_in - mean_no) / mean_in * 100).round(2)
    diff_pct.index = diff_pct.index.astype(int)
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

plot_yearly_trend(df_o3_daily, df_o3_no_daily, 'O3', save_path='/data02/dongnam/output_fig/hadong0/')
plot_yearly_trend(df_no2_daily, df_no2_no_daily, 'NO2', save_path='/data02/dongnam/output_fig/hadong0/')
# %% 2-1. 바람장미 -> 음...........
## 오존 daily outlier 65.20549242424244 // 이산화질소 daily outlier 18.85606060606061
def plot_windrose(df, pol, season, outlier = False, region_name='하동', save_path='/data02/dongnam/output_fig/hadong0/'):
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
    
    pol_name = POLLUTANT(pol).name

    if outlier == True:
        if pol == 'O3':
            bins = [65, 70, 75, 80, 85]
        elif pol == 'NO2':
            bins = [18, 20, 22, 24, 26]
        ax.bar(df['WD'], df[pol], normed=True, opening=0.8, bins=bins, alpha=0.8)
        plt.title(f'{region_name} {season} 풍향별 {pol_name} 고농도 분포({len(df)})', fontsize=13)
        ax.set_yticks([8,16,24,32,40])
        ax.set_yticklabels(['8%','16%','24%','32%','40'], fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=7)
        plt.tight_layout()
        plt.savefig(f'{save_path}{pol}_{season}_고농도_windrose.png', dpi=300)
        plt.show()
    else:
        plt.title(f'{region_name} {season} 풍향별 {pol_name} 농도 분포{len(df)}', fontsize=13)
        ax.set_yticks([5, 10, 15, 20, 25])
        ax.set_yticklabels(['5%','10%','15%','20%','25%'], fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=7)
        plt.tight_layout()
        plt.savefig(f'{save_path}{pol}_{season}_windrose.png', dpi=300)
        plt.show()

#plot_windrose(outliers_o3_daily, 'O3', '겨울', outlier=True)
plot_windrose(outliers_no2_daily, 'NO2', '겨울', outlier=True)
