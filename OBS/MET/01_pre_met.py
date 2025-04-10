#%%
import pandas as pd

year = '2024'
tot_df = pd.DataFrame()
# %%
months = [str(i).zfill(2) for i in range(1, 13)]
for i in months:
    df = pd.read_csv(f'/home/smkim/materials/ASOS/raw/{year}{i}.csv')
    tot_df = pd.concat([tot_df,df],ignore_index=True)
#%%    
df_sorted = tot_df.sort_values(by='stnId', ascending=True)

#%%
ori_stn_info = pd.read_csv(f'./ASOS_info.csv',encoding='cp949')
stn_info = pd.read_csv(f'./ASOS_info.csv',encoding='cp949')
year = pd.to_datetime(year).year  # 삭제할 기준이 되는 연도
stn_info['종료일'].fillna(pd.to_datetime('2025-12-31'), inplace=True)
stn_info = stn_info[pd.to_datetime(stn_info['시작일']).dt.year < year]
stn_info = stn_info[pd.to_datetime(stn_info['종료일']).dt.year > year]

#%%
stn_info['위도'] = stn_info['위도'].round(3)
stn_info['경도'] = stn_info['경도'].round(3)
stn_info['stnId'] = stn_info['지점']
# %%
df_merged = pd.merge(df_sorted, stn_info[['stnId', '위도', '경도']], on='stnId', how='left')
df_merged = df_merged.dropna(subset=['위도'])

# %%
df = pd.DataFrame()
def combine_columns(row):
    return f"{row['stnId']}_{row['경도']}_{row['위도']}"

df['id'] = df_merged.apply(combine_columns, axis=1)
df['tm'] = df_merged['tm']
#%%
for var in ['ta','ws','wd','rn','icsr','ss','ps','pa','td','pv','hm']:
    df[var] = df_merged[var]
    pivot_df = df.pivot(index='tm', columns='id', values=var)
    pivot_df.to_csv(f'/data01/OBS/met_orig/ASOS/data/{year}_{var}.csv')
    print('Success',var)
#%%

