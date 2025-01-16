#%%
#station에 대한 2016-2022 daily trend

import matplotlib.pyplot as plt
import pandas as pd

file_path = '/home/intern/data/pollutant/all_PM25.csv'
df = pd.read_csv(file_path)

df['측정일시'] = pd.to_datetime(df['측정일시'], format='%Y%m%d%H', errors='coerce')

fig, axs = plt.subplots(2, 1, figsize=(30, 10))

for i, ax in zip(df.columns[1:3], axs.flatten()):
    info = i
    split_info = info.split('_')
    std = split_info[0]
    lon = split_info[1]
    lat = split_info[2]

    ax.plot(df['측정일시'], df[info], color='orange', linestyle='-')
    ax.set_title('PM$_{2.5}$ Timeseries of station %s' % (std))
    ax.set_xlabel('Month')
    ax.set_ylabel('μm/m$^3$')
    ax.grid(True)

plt.tight_layout()
plt.savefig('/home/intern/shjin/all_PM25_subplots.png')
plt.show()
#%%
#station에 대한 2016-2022 monthly trend

import matplotlib.pyplot as plt
import pandas as pd

file_path = '/home/intern/data/pollutant/all_PM25.csv'
df = pd.read_csv(file_path)

df['측정일시'] = pd.to_datetime(df['측정일시'], format='%Y%m%d%H', errors='coerce')

df_monthly = df.resample('M', on='측정일시').mean()
fig, axs = plt.subplots(3, 1, figsize=(30, 10))

for i, ax in zip(df.columns[1:4], axs.flatten()):
    info = i
    split_info = info.split('_')
    std = split_info[0]
    lon = split_info[1]
    lat = split_info[2]

    ax.plot(df_monthly.index, df_monthly[info], color='orange', linestyle='-')
    ax.set_title('PM$_{2.5}$ Timeseries of station %s' % (std))
    ax.set_xlabel('Month')
    ax.set_ylabel('μm/m$^3$')
    ax.grid(True)

plt.tight_layout()
plt.savefig('/home/intern/shjin/all_PM25_subplots.png')
plt.show()

#%%
# 여러개의 지점의 연도별 평균 Trend

import pandas as pd
import matplotlib.pyplot as plt

file_path = '/home/intern/data/pollutant/all_PM25.csv'

df = pd.read_csv(file_path, sep=',')
df['측정일시'] = pd.to_datetime(df['측정일시'], format='%Y%m%d%H', errors='coerce')

df_yearly = df.resample('Y', on='측정일시').mean()
df_yearly.index = df_yearly.index.strftime('%Y')

fig, ax = plt.subplots(figsize=(10,5))
for i in df.columns[1:10]:
    info = i
    split_info = info.split('_')
    std = split_info[0]
    lon = split_info[1]
    lat = split_info[2]
    ax.plot(df_yearly.index, df_yearly[info], '-o', label=info[:6])
    ax.legend(loc='upper right', fontsize=10)

ax.set_title('Connecticut PM$_{2.5}$ Monthly Design Value Trends')
ax.set_xlabel('Year', fontsize=13)
ax.set_ylabel('PM$_{2.5}$ [$\mu/m^3$]')

plt.savefig('/home/intern/shjin/Connecticut PM25 Monthly Annual Design Value Trends.png')
plt.tight_layout()

#%%
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df_yearly.index, df_yearly[info], '-o', label=std)
ax.set_xlabel('Year', fontsize=13)
ax.set_ylabel('PM$_{2.5} [$\mu$/m$^3$]')
ax.legend(loc='upper right', fontsize=13)

# %%
