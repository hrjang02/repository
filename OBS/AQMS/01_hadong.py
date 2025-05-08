#%%
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from AQMS_TOOL import convert_24_to_00
import numpy as np
import seaborn as sns
from scipy.stats import iqr
from shapely.geometry import Point
from air_toolbox import plot
from air_toolbox.config import kor_shp_path
import geopandas as gpd
#%%
aws_path = '/data02/dongnam/met/AWS/raw/*'
aqms_path = '/data02/dongnam/data/*.csv'

# AWS, AQMS 데이터 불러오기
years = [str(i) for i in range(2019, 2024)]
aws_files = [f for f in glob(aws_path) if any(year in f for year in years)]
aws_data = pd.concat([pd.read_csv(f) for f in aws_files], ignore_index=True)
aqms_files = [f for f in glob(aqms_path) if any(year in f for year in years)]
aqms_data = pd.concat([pd.read_csv(f) for f in aqms_files], ignore_index=True)

# AWS 데이터 전처리
'''하동읍(932)'''
aws = aws_data[aws_data['STN'].astype(str).str.contains('932')][['KST', 'STN', 'WD', 'WS']]
aws['KST'] = pd.to_datetime(aws['KST'], format='%Y%m%d%H%M')
aws = aws.groupby(['KST']).agg({'WD': 'mean', 'WS': 'mean'}).reset_index()
aws = aws.drop(aws.index[0])
aws['KST'] = aws['KST'].astype(str).apply(convert_24_to_00)
#%% aqms 데이터 전처리
def preprocess_aqms(data, region_filter=None, station_filter=None):
    if region_filter:
            data = data[data['지역'].str.contains(region_filter)]

    if station_filter:
        data = data[data['측정소코드'].isin(station_filter)]
    
    data = data[['측정소코드', '측정일시', 'SO2']]
    data = data.rename(columns={'측정일시': 'KST','측정소코드': 'STN','SO2': 'SO2'})
    
    data['KST'] = data['KST'].astype(str).apply(convert_24_to_00)
    data['KST'] = pd.to_datetime(data['KST'], format='%Y%m%d%H')
    data = data.groupby(['KST']).agg({'SO2': 'mean'}).reset_index()

    data[['SO2']] = data[['SO2']] * 1000
    
    data['Year'] = data['KST'].dt.year
    data['Month'] = data['KST'].dt.month
    
    return data

aqms = preprocess_aqms(aqms_data, station_filter=[238161, 238162])
ulsan_aqms = preprocess_aqms(aqms_data, region_filter='울산')
ulsan_aqms = ulsan_aqms.loc[ulsan_aqms['KST'] < '2024-01-01']
dongnam_aqms = preprocess_aqms(aqms_data, region_filter='부산|대구|진주|고성|창원|김해|양산|구미|칠곡|경산|영천|포항|경주')
#%% 하동 AWS, AQMS 데이터 병합
aqms['KST'] = pd.to_datetime(aqms['KST'])
aws['KST'] = pd.to_datetime(aws['KST'])
df = pd.merge(aqms, aws, on='KST', how='inner')
df['KST'] = pd.to_datetime(df['KST'])
df['Year'] = df['KST'].dt.year
df['Month'] = df['KST'].dt.month
df['Hour'] = df['KST'].dt.hour
df = df.set_index('KST')
df=df[['SO2','WD','Year','Month','Hour']]
df = df.dropna()
#%% 하동 카테고리 나누기
conditions = [
    ((df['WD'] >= 0) & (df['WD'] < 90)),
    (df['WD'] >= 90) & (df['WD'] < 180),
    (df['WD'] >= 180) & (df['WD'] < 270),
    ((df['WD'] >= 270) & (df['WD'] <= 360))
]
choices = ['1', '2', '3', '4']
df['category'] = np.select(conditions, choices, default=np.nan)
#%% 하동 / 울산 outlier
# 하동
Q1 = df['SO2'].quantile(0.25)
Q2 = df['SO2'].quantile(0.50)
Q3 = df['SO2'].quantile(0.75)
IQR_value = iqr(df['SO2'])

lower_bound = Q1 - 1.5 * IQR_value
upper_bound = Q3 + 1.5 * IQR_value

outliers = df[(df['SO2'] > upper_bound)]
# df_in = outlier 포함한 친구
# df_no = outlier 제거한 친구
df_no = df[~df['SO2'].isin(outliers['SO2'])]
df_in = df.copy()

# 울산 outlier
ulsan_Q1 = ulsan_aqms['SO2'].quantile(0.25)
ulsan_Q3 = ulsan_aqms['SO2'].quantile(0.75)
IQR_value = iqr(ulsan_aqms['SO2'])

lower = ulsan_Q1 - 1.5 * IQR_value
upper = ulsan_Q3 + 1.5 * IQR_value

ulsan_outlier = ulsan_aqms[(ulsan_aqms['SO2'] > upper)]

df_ulsan_no = ulsan_aqms[~ulsan_aqms['SO2'].isin(ulsan_outlier['SO2'])]
df_ulsan_in =ulsan_aqms.copy()
#%% 1-0. boxplot
fig, ax = plt.subplots(figsize=(4.5, 5))
df.boxplot(column='SO2', ax=ax)

plt.yscale('log')

ax.text(1.1, Q3, f'Q3 = {Q3:.2f}', va='center', fontsize=10, color='blue', fontweight='bold')
ax.text(0.9, Q2, f'Q2 (Median) = {Q2:.2f}', va='center', ha='right', fontsize=10, color='green', fontweight='bold')
ax.text(1.1, Q1, f'Q1 = {Q1:.2f}', va='center', fontsize=10, color='blue', fontweight='bold')
ax.text(1.1, lower_bound+0.5, f'Lower bound = {lower_bound:.2f}', va='center', fontsize=10, color='red', fontweight='bold')
ax.text(1.1, upper_bound, f'Upper bound = {upper_bound:.2f}', va='center', fontsize=10, color='red', fontweight='bold')

for yval in [Q1, Q2, Q3, lower_bound, upper_bound]:
    ax.axhline(y=yval, color='gray', linestyle='--', linewidth=0.5)

ax.set_title('하동 SO$_2$ Boxplot', fontsize=14)
ax.set_ylabel('SO$_2$ (ppb)')
ax.set_xticklabels([f'전체 데이터({len(df)})'])

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig('/home/hrjang2/0_code/hadong/하동_boxplot.png', dpi=300)
plt.show()
#%% 1-0. 하동 vs 울산 vs 동남 annual mean
h_mean = df.groupby('Year')['SO2'].mean().drop(2024, errors='ignore')
u_mean = ulsan_aqms.groupby('Year')['SO2'].mean().drop(2024, errors='ignore')
d_mean = dongnam_aqms.groupby('Year')['SO2'].mean().drop(2024, errors='ignore')

plt.figure(figsize=(7, 3))
plt.plot(h_mean.index, h_mean.values, marker='o', label='하동', color = 'red')
plt.plot(u_mean.index, u_mean.values, marker='o', label='울산', color = 'blue')
plt.plot(d_mean.index, d_mean.values, marker='o', label='동남권', color = 'black')
plt.axhline(y=3.0, color='red', linestyle='--', linewidth=1)

plt.title('연도별 SO$_2$ 평균 농도', fontsize=14)
plt.xlabel('연도', fontsize=12)
plt.ylabel('SO$_2$ 농도 (ppb)', fontsize=12)
plt.xticks(h_mean.index)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
# plt.savefig('/home/hrjang2/0_code/hadong/yearly_SO2.png', dpi=300)
plt.show()
#%% 1-1. 울산 하동 outlier 포함/제외 mean median 비교
h_mean_in  = df_in['SO2'].mean()
h_mean_no  = df_no['SO2'].mean()
h_med_in   = df_in['SO2'].median()
h_med_no   = df_no['SO2'].median()

u_mean_in  = df_ulsan_in['SO2'].mean()
u_mean_no  = df_ulsan_no['SO2'].mean()
u_med_in   = df_ulsan_in['SO2'].median()
u_med_no   = df_ulsan_no['SO2'].median()

print("📊 SO₂ 통계 요약 (단위:ppb)")
print("-" * 70)
print(f"{'구분':<10}{'하동 Mean':>12}{'울산 Mean':>12}{'하동 Median':>14}{'울산 Median':>14}")
print("-" * 70)
print(f"{'고농도 포함':<10}{h_mean_in:>12.3f}{u_mean_in:>12.3f}{h_med_in:>14.3f}{u_med_in:>14.3f}")
print(f"{'고농도 제외':<10}{h_mean_no:>12.3f}{u_mean_no:>12.3f}{h_med_no:>14.3f}{u_med_no:>14.3f}")
print("-" * 70)
#%% 1-2. histogram plot
'''
histogram plot
'''

plt.figure(figsize=(5,3))

# 고농도 포함
sns.histplot(data=df_in, x='SO2', bins=30, element='step', stat='count',
             common_norm=False, color='gray', label='고농도 포함')

# 고농도 제외
sns.histplot(data=df_no, x='SO2', bins=30, element='step', stat='count',
             common_norm=False, color='skyblue', label='고농도 제외')

plt.title('하동 SO$_2$ 분포', fontsize=14)
plt.xlabel('SO$_2$ 농도 (ppb)', fontsize=12)
plt.ylabel('frequency(log)', fontsize=12)
#plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('/home/hrjang2/0_code/hadong/하동_histogram_SO2_NOTlog.png', dpi=300)
plt.show()
#%% 1-3. 하동 outlier 포함 /제외 농도 
mean_in = df_in.groupby('Year')['SO2'].mean()
mean_no = df_no.groupby('Year')['SO2'].mean()
years = mean_in.index.astype(int).tolist()
with_outlier = mean_in.values
without_outlier = mean_no.values

ulsan_in = df_ulsan_in.groupby('Year')['SO2'].mean()
ulsan_no = df_ulsan_no.groupby('Year')['SO2'].mean()
ulsan_mean_in = ulsan_in.values
ulsan_mean_no = ulsan_no.values

diff_pct = ((mean_in - mean_no) / mean_in * 100).round(2)
diff_ulsan_pct = ((ulsan_in - ulsan_no) / ulsan_in * 100).round(2)

plt.figure(figsize=(6, 3))
plt.plot(years, with_outlier, marker='o', label='고농도 포함', color = 'red')
plt.plot(years, without_outlier, marker='o', label='고농도 제외', color = 'black')
plt.title('SO$_2$ 평균 농도', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('SO$_2$ 평균 농도 (ppb)', fontsize=12)
plt.xticks(years)  
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
# plt.savefig('/home/hrjang2/0_code/hadong/하동_Yearly_SO2.png', dpi=300)
plt.show()

# 하동 증가율

# ax = diff_pct.plot(kind='bar', color='skyblue', figsize=(5, 3))
# plt.title('SO$_2$ 농도 증가율 (%)', fontsize=12)
# plt.ylabel('Difference (%)')
# plt.xlabel('Year')
# plt.xticks(rotation=0)
# plt.grid(True, linestyle='--', alpha=0.5, axis='y')

# for i, v in enumerate(diff_pct.values):
#     ax.text(i, v / 2, f'{v:.1f}%', ha='center', va='center', color='black', fontsize=4)

# plt.tight_layout()
# # plt.savefig('/home/hrjang2/0_code/hadong/하동_Yearly_SO2_growthrate.png', dpi=300)
# plt.show()

x = np.arange(len(diff_pct.index))
width = 0.4

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(x - width/2, diff_pct.values, width, label='하동', color='skyblue')
ax.bar(x + width/2, diff_ulsan_pct.values, width, label='울산', color='orange', alpha=0.7)

for i, v in enumerate(diff_pct.values):
    ax.text(i - width/2, v -2, f'{v:.1f}%', ha='center', va='bottom', color='black', fontsize=8)

for i, v in enumerate(diff_ulsan_pct.values):
    ax.text(i + width/2, v -2, f'{v:.1f}%', ha='center', va='bottom', color='black', fontsize=8)

plt.title('SO$_2$ 농도 증가율 (%)', fontsize=12)
plt.ylabel('Difference (%)')
plt.xlabel('Year')
plt.xticks(rotation=0)
ax.set_xticks(x)
ax.set_xticklabels(diff_pct.index)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.legend()
plt.tight_layout()
# plt.savefig('/home/hrjang2/0_code/hadong/울산하동_Yearly_SO2_growthrate.png', dpi=300)
plt.show()
#%% 2-1. 농도 바람장미
var = 'SO2'
custom_bins = {
    'SO2':  [0, 4.5, 10, 15, 20, 25],
    'NO2':  [0, 8, 16, 24, 32],
    'CO':   [0, 200, 400, 600, 800],
    'O3':   [0, 20, 40, 60, 80],
    'PM10': [0, 10, 20, 30, 40, 50],
    'PM25': [0, 8, 16, 24, 32, 40],
}
bins = custom_bins.get(var)

df_spring = df.loc[(df['Month'] >= 3) & (df['Month'] <= 5)]
df_summer = df.loc[(df['Month'] >= 6) & (df['Month'] <= 8)]
df_fall = df.loc[(df['Month'] >= 9) & (df['Month'] <= 11)]
df_winter = df.loc[(df['Month'] == 12) | (df['Month'] <= 2)]


fig = plt.figure(figsize=(6, 5))
ax = WindroseAxes.from_ax(fig=fig)
ax.bar(df['WD'], df[var], normed=True, opening=0.8, bins=bins, alpha=0.8)
plt.title(f'하동(#932) 풍향별 SO$_2$ 농도 분포', fontsize=13)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=7)

ax.set_yticks([5, 10, 15, 20, 25])
ax.set_yticklabels(['5%','10%','15%','20%', '25%'], fontsize=12)
plt.tight_layout()
plt.savefig('/home/hrjang2/0_code/hadong/하동(#932)_windrose.png', dpi=300, bbox_inches='tight')
plt.show()
#%% 2-2. 카테고리별 히스토그램
thresholds = [0, 5, 10, 15, 20, 25]  # 농도 구간
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '25+']  # 농도 구간 레이블
colors = ['#ff9999', '#ffcc80', '#a3d9a5', '#99ccff', '#c2a5d8']

bins_df = pd.DataFrame(0, index=['1', '2', '3', '4'], columns=labels)

# for cat in ['1', '2', '3', '4']:
#     sub = df[df['category'] == cat]
#     sub_out = sub[sub[var] >= thresholds[0]]  
#     for i in range(len(thresholds)):
#         lower = thresholds[i]
#         upper = thresholds[i+1] if i+1 < len(thresholds) else np.inf
#         count = sub_out[(sub_out[var] >= lower) & (sub_out[var] < upper)].shape[0]
#         bins_df.loc[cat, labels[i]] = count

# fig, ax = plt.subplots(figsize=(9, 4))
# bottom = np.zeros(len(bins_df))
# x = np.arange(len(bins_df))

# for i, label in enumerate(labels):
#     values = bins_df[label].values
#     ax.bar(x, values, bottom=bottom, label=label + 'ppb', color=colors[i])
#     bottom += values  

# for idx, xpos in enumerate(x):
#     values_text = ' / '.join(str(int(bins_df[label].values[idx])) for label in labels)
#     y_pos = bottom[idx] -200
#     ax.text(xpos, y_pos, values_text, ha='center', va='bottom', fontsize=10)
# ax.set_xticks(x)
# ax.set_xticklabels(['1(0~90)', '2(90~180)', '3(180~270)', '4(270~360)'], fontsize=12)
# ax.set_xlabel('풍향 카테고리', fontsize=14)
# ax.set_ylabel('outlier count', fontsize=14)
# ax.set_title(f'하동 {var} 고농도 농도별 누적 분포', fontsize=16)
# ax.grid(True, linestyle='--', axis='y', alpha=0.7)
# ax.legend(title='농도 구간', fontsize=8)
# plt.tight_layout()
# plt.savefig('/home/hrjang2/0_code/hadong/하동(#932)_category_histogram.png', dpi=300, bbox_inches='tight')
# plt.show()

thresholds = [0, 4.5, 10, 15, 20, 25]
labels = ['0-4.5', '4.5-10', '11-15', '16-20', '21-25', '25+']
colors = ['#ff9999', '#ffcc80', '#a3d9a5', '#99ccff', '#c2a5d8', '#f4b183']  # (6개 맞춤)

bins_df = pd.DataFrame(0, index=['1', '2', '3', '4'], columns=labels)
bins_df.index.name = 'category'
bins_df.columns.name = 'label'

for cat in ['1', '2', '3', '4']:
    sub = df[df['category'] == cat]
    for i in range(len(thresholds)):
        lower = thresholds[i]
        upper = thresholds[i+1] if i+1 < len(thresholds) else np.inf
        count = sub[(sub[var] >= lower) & (sub[var] < upper)].shape[0]
        bins_df.loc[cat, labels[i]] = count

print(bins_df)
#%% 2-2. 계절별 히스토그램
def get_season(month):
    if month in [3, 4, 5]:
        return '봄'
    elif month in [6, 7, 8]:
        return '여름'
    elif month in [9, 10, 11]:
        return '가을'
    else:
        return '겨울'

df['Season'] = outliers.index.month.map(get_season)

seasonal_counts = outliers.groupby(['Season', 'category']).size().reset_index(name='Count')
season_totals = seasonal_counts.groupby('Season')['Count'].transform('sum')
seasonal_counts['Percent'] = seasonal_counts['Count'] / season_totals * 100

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
    percents = cat_data['Percent'].values

    bars = ax.bar(x, counts, bar_width, bottom=bottom, label=str(cat), color=colors[i])

    for j, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bottom[j] + height/2,
                    f'{percents[j]:.1f}%', ha='center', va='center', fontsize=10)

    bottom += counts
season_counts_dict = seasonal_counts.groupby('Season')['Count'].sum().reindex(seasons).fillna(0).astype(int).to_dict()
xtick_labels = [f'{season}({season_counts_dict[season]})' for season in seasons]

ax.set_xticks(x)
ax.set_xticklabels(xtick_labels, fontsize=12)
ax.set_ylabel('Count')
ax.set_title('하동(#932) 계절별 고농도 SO$_2$ 카테고리 분포')
ax.legend(title='category')
plt.tight_layout()
plt.savefig('/home/hrjang2/0_code/hadong/하동(#932)_계절별_category_histogram.png', dpi=300)
plt.show()
# %% 2-3. 하동 map 만들기
city = '하동'

AWS_path = '/data02/dongnam/met/AWS/data/AWS_stn_info.csv'
geo_path = '/data02/Map_shp/TL_SCCO_SIG_WGS.shp'

AWS_info = pd.read_csv(AWS_path,encoding='cp949')
AWS_info['geometry'] = [Point(xy) for xy in zip(AWS_info['경도'], AWS_info['위도'])]
AWS_gdf = gpd.GeoDataFrame(AWS_info, geometry='geometry')  # WGS 84 좌표계

dn_shp = plot.regional_shp('Dongnam')
AWS = gpd.sjoin(AWS_gdf, dn_shp, how='inner')

AWS['시작일'] = pd.to_datetime(AWS['시작일'], format='%Y-%m-%d')
AWS['종료일'] = pd.to_datetime(AWS['종료일'], format='%Y-%m-%d', errors='coerce')
AWS = AWS[~ (AWS['종료일'] <= '2024-01-01')]
# AWS = AWS[['지점', '지점명', '지점주소', '위도', '경도']]

aqms_data = pd.DataFrame({
    '지점명': ['하동읍', '금성면'],
    '경도': [127.7514, 127.7943],
    '위도': [35.0677, 34.9659]
})
aqms_data['geometry'] = aqms_data.apply(lambda row: Point(row['경도'], row['위도']), axis=1)
AQMS_gdf = gpd.GeoDataFrame(aqms_data, geometry='geometry', crs='EPSG:4326')
AQMS_city = AQMS_gdf

danger_data = pd.DataFrame({
    '지점명': ['광양국가산업단지', '하동화력발전소'],
    '경도': [127.758678, 127.820832],
    '위도': [34.938616, 34.950827]
})
danger_data['geometry'] = danger_data.apply(lambda row: Point(row['경도'], row['위도']), axis=1)
danger_gdf = gpd.GeoDataFrame(danger_data, geometry='geometry', crs='EPSG:4326')
danger_name = danger_gdf

def filter_city_stations(gdf, kor_shp_filtered):
    joined = gpd.sjoin(gdf, kor_shp_filtered , how='inner')
    joined['시작일'] = pd.to_datetime(joined['시작일'], format='%Y-%m-%d')
    joined['종료일'] = pd.to_datetime(joined['종료일'], format='%Y-%m-%d', errors='coerce')
    joined = joined[~(joined['종료일'] <= '2024-01-01')]
    return joined

map_fname = f'{kor_shp_path}/TL_SCCO_SIG_WGS_utf8.shp'
kor_shp = gpd.read_file(map_fname)

kor_shp_filtered = kor_shp[kor_shp['SIG_KOR_NM'].isin(['하동군', '광양시'])]
AWS_city = filter_city_stations(AWS_gdf, kor_shp[kor_shp['SIG_KOR_NM'].isin(['하동군'])])

fig, ax = plt.subplots(figsize=(9, 8))
ax.set_title(f'{city}', fontsize=20)
kor_shp_filtered.plot(ax=ax, color='lightgray', edgecolor='gray')

ax.set_xlim(127.6, 128.0)
ax.set_ylim(34.85, 35.15)

# AWS
for i, (_, row) in enumerate(AWS_city.iterrows()):
    label = 'AWS' if i == 0 else None
    ax.scatter(row['경도'], row['위도'], c='black', s=20, label=label)
    ax.text(row['경도']+0.001, row['위도'] - 0.015, row['지점명'],
            fontsize=15, ha='center', c='black', fontweight='bold')
# AQMS
for i, (_, row) in enumerate(AQMS_city.iterrows()):
    label = 'AQMS' if i == 0 else None
    ax.scatter(row['경도'], row['위도'], c='blue', s=20, label=label)
    ax.text(row['경도']-0.003, row['위도'] + 0.012, row['지점명'],
            fontsize=15, ha='center', c='blue', fontweight='bold')
# Danger
for i, (_, row) in enumerate(danger_name.iterrows()):
    label = '주요배출원' if i == 0 else None
    ax.scatter(row['경도'], row['위도'], c='red', s=50, label=label)
    ax.text(row['경도']+0.001, row['위도'] - 0.025, row['지점명'],
            fontsize=15, ha='center', c='red', fontweight='bold')
ax.legend(fontsize=15, loc='upper right')
plt.tight_layout()
# fig.savefig(f'/home/hrjang2/0_code/hadong/하동_map.png', dpi=300)
plt.show()
# %%
