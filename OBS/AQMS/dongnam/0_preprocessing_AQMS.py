# 해야하는거
# 24시간 데이터를 00시간 데이터로 바꾸기(했듬)
# 지자체 별 물질별 분류
# classify_AQMS 이거 사용
#%%
import pandas as pd
#%%
def convert_24_to_00(datetime_str):
    '''Convert 24:00 to 00:00 for the next day

    Args:
        datetime_str (str): datetime string to convert (format: YYYYMMDDHH)

    Return:
        datetime_str (str): converted datetime string
    '''
    
    if datetime_str.endswith("24"):
        date_part = datetime_str[:-2]
        next_date = pd.to_datetime(date_part, format='%Y%m%d') + pd.Timedelta(days=1)
        return next_date.strftime('%Y%m%d00')
    return datetime_str
#%%
def combine_to_year(year):
    '''Read AQMS EXCEL files for each month and combine them into a single DataFrame for the year.

    Args:
        year (int): year to combine
        save (bool): whether to save the combined data as a CSV file(Default: True)

    Return:
        AQMS_combined (pd.DataFrame): combined AQMS data for the year
    '''

    AQMS_path = '/data02/dongnam/data/rawdata'
    all_months_data = []

    for month in range(1, 13):
        file_path = AQMS_path + f'/{year}년 {month}월.xlsx'
        try:
            month_data = pd.read_excel(file_path)
            all_months_data.append(month_data)
        except FileNotFoundError:
            print(f'No data for {year} {month}')
            continue
    
    # Combine all month's data into a single DataFrame
    AQMS_combined = pd.concat(all_months_data, ignore_index=True)

    # Convert pollutant data (SO2, CO, O3, NO2) to µg/m³ by multiplying by 1000
    AQMS_combined[['SO2', 'CO', 'O3', 'NO2']] *= 1000

    return AQMS_combined
#%%
def classify_AQMS(year, output_path = '/data01/Share/from_hrjang'):
    '''Classify AQMS data by pollutant and region and save them as CSV files

    Args:
        year (int): year to classify
        output_path (str): path to save (Default: '/data01/Share/from_hrjang/AQMS')
    '''
    pollutants = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
    
    AQMS_data = combine_to_year(year)

    # Convert '측정일시' to datetime format
    AQMS_data['측정일시'] = AQMS_data['측정일시'].astype(str).apply(convert_24_to_00)
    AQMS_data['측정일시'] = pd.to_datetime(AQMS_data['측정일시'], format='%Y%m%d%H')

    # Iterate over pollutants
    for pollutant in pollutants:
        AQMS_classify = AQMS_data[['측정일시', '측정소코드', pollutant]]
        region_data = AQMS_classify.pivot_table(index='측정일시', columns='측정소코드', values=pollutant).reset_index()

        # # Save the file
        file_path = f'{output_path}/{year}_{pollutant}.csv'
        region_data.to_csv(file_path, index=False)
        print(f"Saved {file_path}")
# %%
