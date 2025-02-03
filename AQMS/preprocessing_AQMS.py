# 해야하는거
# 24시간 데이터를 00시간 데이터로 바꾸기(했듬)
# 지자체 별 물질별 분류
#%%
import pandas as pd
# %%
class Region:
    def __init__(self, region):
        region_info = {
            '서울' : {'name':'Seoul'}, '부산' : {'name':'Busan'}, '대구' : {'name':'Daegu'}, '인천' : {'name':'Incheon'},
            '광주' : {'name':'Gwangju'}, '대전' : {'name':'Daejeon'}, '울산' : {'name':'Ulsan'}, '세종' : {'name':'Sejong'},
            '경기' : {'name':'Gyeonggi'}, '강원' : {'name':'Gangwon'}, '충북' : {'name':'Chungbuk'}, '충남' : {'name':'Chungnam'},
            '전북' : {'name':'Jeonbuk'}, '전남' : {'name':'Jeonnam'}, '경북' : {'name':'Gyeongbuk'}, '경남' : {'name':'Gyeongnam'},
            '제주' : {'name':'Jeju'}
        }
        self.region = region_info[region]['name']
#%%
def convert_24_to_00(datetime):
    '''Convert 24:00 to 00:00 for the next day

    Args:
        datetime (str): datetime string to convert

    Return:
        datetime (str): converted datetime string
    '''

    if datetime.endswith("24"):
        date_part = datetime[:-2]
        next_date = pd.to_datetime(date_part, format='%Y%m%d') + pd.Timedelta(days=1)
        return next_date.strftime('%Y%m%d00')
    return datetime
#%%
def combine_to_year(year):
    '''Read AQMS EXCEL files for each month and combine them into a single DataFrame for the year.
       Automatically converts 24:00 to 00:00 in the '측정일시' column
        Save the combined data as a CSV file if save is True

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

            # Convert 24:00 to 00:00 if '측정일시' column exists
            month_data['측정일시'] = month_data['측정일시'].astype(str)
            month_data['측정일시'] = month_data['측정일시'].apply(convert_24_to_00)
            month_data['측정일시'] = pd.to_datetime(month_data['측정일시'], format='%Y%m%d%H')

            # Convert pollutant data (SO2, CO, O3, NO2) to µg/m³ by multiplying by 1000
            month_data[['SO2', 'CO', 'O3', 'NO2']] *= 1000

            all_months_data.append(month_data)

        except FileNotFoundError:
            print(f'No data for {year} {month}')
            continue
    
    # Combine all month's data into a single DataFrame
    AQMS_combined = pd.concat(all_months_data, ignore_index=True)
    
    # Save the combined data to CSV
    AQMS_combined.to_csv(f'/data01/Share/from_hrjang/{year}_AQMS.csv', index=False)

    return AQMS_combined
#%%
def classify_AQMS(AQMS_data, output_path = '/data01/Share/from_hrjang/AQMS'):
    '''Classify AQMS data by pollutant and region and save them as separate files

    Args:
        AQMS_data (pd.DataFrame): AQMS data to classify
        output_path (str): path to save the classified data(Default: '/data01/Share/from_hrjang/AQMS')

    Return:
        None
    '''

    pollutants = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
    regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
    
    AQMS_data['측정일시'] = pd.to_datetime(AQMS_data['측정일시'], errors='coerce')
    year = AQMS_data['측정일시'].dt.year.unique()[0]

    # Iterate over pollutants and regions
    for pollutant in pollutants:
        for region in regions:
            try:
                # Filter by region and classify the data
                region_classify = AQMS_data[AQMS_data['지역'].str.split(' ').str[0] == region]
                region_classify = region_classify[['측정일시', '측정소코드', pollutant]]
                region_data = region_classify.pivot_table(index='측정일시', columns='측정소코드', values=pollutant).reset_index()

                # Save the classified data as a CSV
                region_obj = Region(region).region
                region_data.to_csv(f'{output_path}/{year}_{region_obj}_{pollutant}.csv', index=False)

            except KeyError:
                print(f"No data for {region} {pollutant}")
                continue
# %%
