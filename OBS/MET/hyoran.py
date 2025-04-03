#%%
# ASOS만 일단 사용함
# var = 기온(ta), 습도(pa), 강수량(rn), 풍속(ws), 풍향(wd)
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/home/hrjang2/0_code/AQMS')
from AQMS_TOOL import convert_24_to_00, POLLUTANT, DONGNAM
#%%
def load_obs_data(start_year, end_year, data_path='/data02/dongnam/data/AWS_ASOS/'):
    """
    Loads and combines ASOS data for a given range of years.

    Parameters:
    - start_year (int): The starting year of the data range.
    - end_year (int): The ending year of the data range.
    - data_path (str): The base directory path where the CSV files are stored.

    Returns:
    - pd.DataFrame: Combined DataFrame of ASOS data indexed by date.
    """
    data_stn_list = []

    for year in range(start_year, end_year+1):
        file_path = f'{data_path}{year}.csv'
        df_ASOS = pd.read_csv(file_path)
        df_ASOS['일시'] = pd.to_datetime(df_ASOS['일시'])
        df_stn_year = df_ASOS.loc[:, ~df_ASOS.columns.duplicated()]

        data_stn_list.append(df_stn_year)

    df_stn = pd.concat(data_stn_list)
    df_stn = df_stn.set_index('일시')

    return df_stn

#%% Time Series Plot
# Example usage
# plot_time(2019, 2023, var='기온(°C)', freq='D', region='부산')
# plot_time(2019, 2023, var='기온(°C)', freq='D', stn_num='160')

def plot_time(start_year, end_year, var, freq, region=None, stn_num=None):
    '''
    Parameters:
    - start_year: starting year (int)
    - end_year: ending year (int)
    - region: region name (str) 
    - stn_num: station number (str) 
    - var: variable name (str) like '기온(°C)', '풍속(m/s)', '강수량(mm)', '습도(%)'
    - freq: frequency (str) like 'D', 'M', 'Y'
    '''
    df_stn = load_obs_data(start_year, end_year)

    if region is not None:
        df_region = df_stn[df_stn['지점명'].str.contains(region)]
        region = DONGNAM(region).city
    elif stn_num is not None:
        df_region = df_stn[df_stn['지점'] == stn_num]
    else:
        print("Region or station number must be provided.")
        return

    df_resampled = df_region.resample(freq).mean(numeric_only=True)
    
    if var not in df_resampled.columns:
        print(f"'{var}' not found in the data.")
        return

    if var == '기온(°C)':
        var_name = 'Temperature'
    elif var == '풍속(m/s)':
        var_name = 'Wind Speed'
    elif var == '강수량(mm)':
        var_name = 'Precipitation'
    elif var == '습도(%)':
        var_name = 'Humidity'
    else:
        print("Invalid variable name.")
        return
    
    if freq == 'M':
        plt.figure(figsize=(5, 3))
        plt.title(f'{region} Monthly {var_name} {start_year}-{end_year}')
        plt.xlabel('Month')
    elif freq == 'D':
        plt.figure(figsize=(8, 3))
        plt.title(f'{region} Daily {var_name} {start_year}-{end_year}')
        plt.xlabel('Date')
    elif freq == 'Y':
        plt.figure(figsize=(5, 3))
        plt.title(f'{region} Yearly {var_name} {start_year}-{end_year}')
        plt.xlabel('Year')
    
    plt.plot(df_resampled.index, df_resampled[var], linestyle='-', color='k')
    plt.ylabel(var)
    plt.grid(True)
    plt.show()
    
#%%
# Example usage
# load_station_info()

def load_stn_info(file_path='/data02/dongnam/data/AWS_ASOS/latlon.csv'):
    """
    Reads a CSV file, sets the first column as the index,
    renames the index axis to 'stn', and returns the DataFrame.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Processed DataFrame with 'stn' as the index axis.
    """
    try:
        df_info = pd.read_csv(file_path, index_col=0, encoding='ISO-8859-1')
        df_info = df_info.rename_axis('stn')
        return df_info
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
#%%
