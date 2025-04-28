#%%
import os
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
#%%
def safe_request(url, retries=4, delay=5):
    """재시도 가능한 요청 함수"""
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=(10, 30))  # 연결 10초, 응답 30초
            res.raise_for_status()
            return res.text.strip()
        except requests.exceptions.RequestException as e:
            print(f"[{url}] 오류남 {e}//{attempt+1}/{retries}회")
            time.sleep(delay)
    return None
#%%
def download_aws_month(year, month, auth='PD63DIaGQ6--twyGhiOv5A', output_dir='/home/hrjang2/0_code/'):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"AWS_{year}{month:02d}.csv")

    start_date = datetime(year, month, 1)
    end_date = datetime(year + 1, 1, 1) - timedelta(days=1) if month == 12 else datetime(year, month + 1, 1) - timedelta(days=1)

    # 기존 파일 삭제
    if os.path.exists(output_file):
        os.remove(output_file)

    first_file = True
    current_date = start_date

    while current_date <= end_date:
        for hour in range(24):
            try:
                timestamp = current_date.strftime(f"%Y%m%d{hour:02d}00")
                url = f"https://apihub.kma.go.kr/api/typ01/url/awsh.php?tm={timestamp}&help=1&authKey={auth}"

                content = safe_request(url)
                if not content:
                    continue

                lines = content.splitlines()
                lines = [line for line in lines if not line.strip().startswith('#')]  # 주석 제거

                if len(lines) < 2:
                    continue

                with open(output_file, 'a', encoding='utf-8') as f_out:
                    if first_file:
                        f_out.write(lines[0] + '\n')  # 컬럼 헤더
                        first_file = False
                    f_out.writelines(line + '\n' for line in lines[1:])  # 데이터만 추가

                print(f"[진행] {year}-{month:02d} {timestamp} 저장 완료", end='\r')

            except Exception as e:
                print(f"[{year}-{month:02d}] {current_date.strftime('%Y-%m-%d')} {hour:02d}시 오류: {e}")
                time.sleep(5)

        current_date += timedelta(days=1)

    print(f"\n[완료] {output_file} 저장 완료.")


#%%

# year = 2005
# for month in range(1, 13):
#     try:
#         df = pd.read_csv(
#             f'/data02/dongnam/met/AWS/raw/AWS_{year}{month:02d}.csv',
#             encoding='cp949',
#             skiprows=1,
#             header=None,
#         )
#         df = df['0'].str.split(',', expand=True)
#         df.columns = ['KST', 'STN', 'TA', 'WD', 'WS', 'RN_DAY', 'RN_HR1', 'HM', 'PA', 'PS']
#         df.to_csv(f'/data02/dongnam/met/AWS/raw/AWS_{year}{month:02d}.csv', index=False, encoding='cp949')
#     except:
#         print(f"{year}년 {month}월")
#         continue