#%%
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd

#%%
# datetime 만들어요 (24시 처리 함수)~~~~~~~~~~~~~ 
def convert_24_to_00(datetime):
    """
    Convert 24-hour datetime to 00-hour datetime

    input: datetime (%Y%m%d24, str)
    output: datetime (%Y%m%d00, str)
    """
    if datetime.endswith("24"):
        date_part = datetime[:-2]
        next_date = pd.to_datetime(date_part, format='%Y%m%d') + pd.Timedelta(days=1)
        return next_date.strftime('%Y%m%d00')
    return datetime
#%%
def calculate(actual, predicted):
    """
    
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r, _ = pearsonr(actual, predicted)
    r2 = r2_score(actual, predicted)
    me2 = np.abs(predicted-actual)
    mb = np.mean(predicted-actual)
    me = np.mean(me2)
    nmb = (np.sum(predicted - actual) / np.sum(actual)) * 100
    nme = (np.sum(np.abs(predicted - actual)) / np.sum(actual)) * 100
    model_mean = np.mean(predicted)
    obs_mean = np.mean(actual)
    return rmse, r, r2, mb, me, nmb, nme, model_mean, obs_mean
# %%

