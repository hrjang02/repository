#%%
import wind
import pandas as pd
import numpy as np
#%%
year = '2021'
data_path=f'/data01/OBS/met_orig/ASOS/data/'

ws_df = pd.read_csv(data_path + year + '_ws.csv')
wd_df = pd.read_csv(data_path + year + '_wd.csv')

wspd = ws_df.iloc[:, 1:].values.astype(float).reshape(-1)
wdir = wd_df.iloc[:, 1:].values.astype(float).reshape(-1)
#%%
wd_list = []
ws_list = []

for i in range(len(wdir)):
    if np.isnan(wdir[i]) or np.isnan(wspd[i]):
        u, v = np.nan, np.nan 
    else:
        u, v = wind.component(float(wdir[i]), float(wspd[i]))
    
    wd = wind.uv2wd(u, v)
    ws = wind.uv2ws(u, v)

    wd_list.append(wd)
    ws_list.append(ws)

wind_df = pd.DataFrame({'wdir': wdir, 'cal_wd': wd_list, 'wspd': wspd, 'cal_ws': ws_list})
wind_df.loc[wind_df['wdir'] == 360, 'wdir'] = 0
# %%
def calculator(actual, predicted):
    ME2 = np.abs(predicted-actual)
    ME = np.mean(ME2)
    NME = (np.sum(ME2)/np.sum(actual))*100
    MB = np.mean(predicted-actual)
    NMB = (np.sum(predicted-actual)/np.sum(actual))*100
    RMSE = np.sqrt(np.mean((predicted-actual)**2))
    
    # corr,_ = pearsonr(actual, predicted)
    # corr = format(corr,"5.2f")

    ME = format(ME,"5.2f")
    NME = format(NME,"5.2f")
    MB = format(MB,"5.2f")
    NMB = format(NMB,"5.2f")
    RMSE = format(RMSE,"5.2f")
    return MB,ME,NMB,NME,RMSE

print(calculator(wind_df.wdir, wind_df.cal_wd))
# %%
