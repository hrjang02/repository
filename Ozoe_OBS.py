#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# Load data
var ='O3'
years = ['2020', '2021', '2022', '2023']


df_obs = pd.DataFrame()
for year in years:
    OBS_csv = pd.read_csv(f'/home/hrjang2/{year}_O3.csv')
    OBS_csv['측정일시'] = pd.to_datetime(OBS_csv['측정일시'])

    df_obs = pd.concat([df_obs, OBS_csv])

df_obs = df_obs.set_index('측정일시')
df_obs = df_obs.drop('Unnamed: 0', axis=1)

OZONE_OBS = df_obs[(df_obs.index.hour >= 21) | (df_obs.index.hour <= 3)]
OZONE_OBS = OZONE_OBS.groupby(OZONE_OBS.index.hour).mean()
OZONE_OBS = OZONE_OBS.reindex([21, 22, 23, 0, 1, 2, 3])
OZONE_OBS = OZONE_OBS.mean(axis=1)
# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(OZONE_OBS.index, OZONE_OBS, label='OBS')
ax.set_xticks([21, 22, 23, 0, 1, 2, 3])  # Correct x-ticks
ax.set_xticklabels(['21', '22', '23', '00', '01', '02', '03'])  # x-axis labels
ax.set_xlabel('Time', fontsize=15)
ax.set_ylabel('O3 Concentration', fontsize=15)
ax.set_title('(Time) Night Ozone Concentration', fontsize=20, fontweight="bold")
ax.legend(loc='upper right', fontsize=15)
ax.grid(True)
plt.show()


# %%
'''
관측data로만 자정 월별 그래프
'''

fig, ax = plt.subplots(figsize=(10, 5))
for year in years:

    file = pd.read_csv(f'/home/hrjang2/{year}_O3.csv')
    

    file['측정일시'] = pd.to_datetime(file['측정일시'])
    file.set_index('측정일시', inplace=True)
    
    file = file[file.index.hour == 0]

    file = file.drop(columns='Unnamed: 0')

    file_monthly = file.resample('M').mean()
    row_mean = file_monthly.mean(axis=1)
    monthly_avg = row_mean.groupby(row_mean.index.month).mean()
    ax.plot(monthly_avg.index, monthly_avg, label=f'{year}')


ax.set_xticks(range(1, 13))  # x축을 1~12로 설정
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # x축 레이블 설정
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Monthly O3 Concentration',fontsize=15)
ax.set_title('Monthly Average Night Ozone Concentration',fontsize=20,fontweight="bold")
ax.legend(loc='upper right', fontsize=15)
ax.grid(True)

plt.savefig(f'/home/hrjang2/Monthly_Average_Night_Ozone_Concentration.png')

# %%
