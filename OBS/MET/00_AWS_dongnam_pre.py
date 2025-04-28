#%%
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from air_toolbox import plot
from air_toolbox.config import kor_shp_path
#%%
ASOS_path = '/data02/dongnam/met/ASOS/data/ASOS_stn_info.csv'
AWS_path = '/data02/dongnam/met/AWS/data/AWS_stn_info.csv'
geo_path = '/data02/Map_shp/TL_SCCO_SIG_WGS.shp'
#%%
ASOS_info = pd.read_csv(ASOS_path,encoding='cp949')
ASOS_info['geometry'] = [Point(xy) for xy in zip(ASOS_info['경도'], ASOS_info['위도'])]
ASOS_gdf = gpd.GeoDataFrame(ASOS_info, geometry='geometry')  # WGS 84 좌표계

AWS_info = pd.read_csv(AWS_path,encoding='cp949')
AWS_info['geometry'] = [Point(xy) for xy in zip(AWS_info['경도'], AWS_info['위도'])]
AWS_gdf = gpd.GeoDataFrame(AWS_info, geometry='geometry')  # WGS 84 좌표계
# %%
dn_shp = plot.regional_shp('Dongnam')

ASOS = gpd.sjoin(ASOS_gdf, dn_shp, how='inner')
AWS = gpd.sjoin(AWS_gdf, dn_shp, how='inner')
# %%
ASOS['시작일'] = pd.to_datetime(ASOS['시작일'], format='%Y-%m-%d')
ASOS['종료일'] = pd.to_datetime(ASOS['종료일'], format='%Y-%m-%d', errors='coerce')
ASOS = ASOS[~ (ASOS['종료일'] <= '2024-01-01')]
ASOS = ASOS[['지점', '지점명', '지점주소', '위도', '경도']]

AWS['시작일'] = pd.to_datetime(AWS['시작일'], format='%Y-%m-%d')
AWS['종료일'] = pd.to_datetime(AWS['종료일'], format='%Y-%m-%d', errors='coerce')
AWS = AWS[~ (AWS['종료일'] <= '2024-01-01')]
# AWS = AWS[['지점', '지점명', '지점주소', '위도', '경도']]

# %%############################################################################
# AWS.to_csv('/home/hrjang2/0_code/AWS_dongnam.csv', index=False, encoding='cp949')
#%%
'''
city별로 그리기(ASOS, AWS)
'''
city_dict = {
    '포항': ['47111', '47113'],
    '경주': ['47130'],
    '영천': ['47230'],
    '경산': ['47290'],
    '칠곡': ['47850'],
    '구미': ['47190'],
    '하동': ['48850'],
    '창원': ['48121', '48123', '48125', '48127', '48129'],
    '진주': ['48170'],
    '김해': ['48250'],
    '양산': ['48330'],
    '고성': ['48820'],
    '대구': ['27110', '27140', '27170', '27200', '27230', '27260', '27290', '27710'],
    '울산': ['31110', '31140', '31170', '31200', '31710'],
    '부산': ['26110', '26140', '26170', '26200', '26230', '26260', '26290',
           '26320', '26350', '26380', '26410', '26440', '26470', '26500', '26530', '26710']
}
# %%
dongnam_city = ['부산', '울산', '대구', '하동', '진주', '고성', '창원', '김해', '양산', '구미', '칠곡', '경산', '영천', '포항', '경주']

plt.rc('font', family='NanumGothic') 
plt.rcParams['axes.unicode_minus'] = False


def filter_city_stations(gdf, kor_shp_filtered):
    joined = gpd.sjoin(gdf, kor_shp_filtered , how='inner')
    joined['시작일'] = pd.to_datetime(joined['시작일'], format='%Y-%m-%d')
    joined['종료일'] = pd.to_datetime(joined['종료일'], format='%Y-%m-%d', errors='coerce')
    joined = joined[~(joined['종료일'] <= '2024-01-01')]
    return joined

def plot_stations(ax, gdf, color, text_color):
    ax.scatter(gdf.경도, gdf.위도, c=color, s=20)
    for _, row in gdf.iterrows():
        ax.text(row['경도'], row['위도'] - 0.015, row['지점명'],
                fontsize=15, ha='center', c=text_color, fontweight='bold')

map_fname = f'{kor_shp_path}/TL_SCCO_SIG_WGS_utf8.shp'
kor_shp = gpd.read_file(map_fname)

for city in dongnam_city:
    sig_cd_list = city_dict[city]
    kor_shp_filtered = kor_shp[kor_shp['SIG_CD'].isin(sig_cd_list)]

    ASOS_city = filter_city_stations(ASOS_gdf, kor_shp_filtered)
    AWS_city = filter_city_stations(AWS_gdf, kor_shp_filtered)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f'{city}', fontsize=20)
    kor_shp_filtered.plot(ax=ax, color='lightgray', edgecolor='black')

    plot_stations(ax, ASOS_city, color='red', text_color='blue')
    plot_stations(ax, AWS_city, color='red', text_color='black')

    plt.tight_layout()
    fig.savefig(f'/home/hrjang2/0_code/dongnam_city/{city}_map.png', dpi=300)
    plt.show()
# %%
