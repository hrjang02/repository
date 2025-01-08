#%%
import pandas as pd
import openpyxl

year = 2024
AQMS_path = f'/home/hrjang2/AQMS/data/'

all_months_data = []

for month in range(1, 13):
    file_path = AQMS_path + f'{year}년 {month}월.xlsx'
    try:
        month_data = pd.read_excel(file_path)
        month_data['월'] = month
        all_months_data.append(month_data)
    except FileNotFoundError:
        print(f"{file_path} 파일이 존재하지 않습니다.")
        continue

AQMS_combined = pd.concat(all_months_data, ignore_index=True)
output_path = AQMS_path + f'{year}.csv'
AQMS_combined.to_csv(output_path, index=False)

print(f'saved {year} to {output_path}')
# %%
