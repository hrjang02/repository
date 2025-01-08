import numpy as np
import pandas as pd
from datetime import datetime
#%%
months = [f'{month:02}' for month in range(1,13)]
for year in [2021,2022,2023]:
    for month in months:
        
        df = pd.read_csv(f'/data07/WRF-ML/data/{year}{month}_matched.csv')
        df['WSPD10_uwind'] = df['WSPD10']*np.cos(np.deg2rad(270-df['WDIR10']))
        df['WSPD10_vwind'] = df['WSPD10']*np.sin(np.deg2rad(270-df['WDIR10']))

        df['tm'] = pd.to_datetime(df['tm'])
        
        df.to_csv(f'/data07/WRF-ML/data/{year}{month}_matched_uv.csv')