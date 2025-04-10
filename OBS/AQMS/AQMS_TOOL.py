#%%
import pandas as pd
#%% 
def convert_24_to_00(datetime):
    '''
    Convert 24:00 to 00:00
    '''
    if datetime.endswith("24"):
        date_part = datetime[:-2]
        next_date = pd.to_datetime(date_part, format='%Y%m%d') + pd.Timedelta(days=1)
        return next_date.strftime('%Y%m%d00')
    return datetime
#%%
class POLLUTANT:
    def __init__(self, pollutant):
        pollutant_info = {
            'SO2': {'unit': 'ppb', 'name': 'SO$_2$', 'standard': 20},
            'CO': {'unit': 'ppm', 'name': 'CO', 'standard': 1},
            'O3': {'unit': 'ppb', 'name': 'O$_3$', 'standard': 100},
            'NO2': {'unit': 'ppb', 'name': 'NO$_2$', 'standard': 30},
            'PM10': {'unit': '$\\mu g/m^3$', 'name': 'PM$_{10}$', 'standard' : 15},
            'PM25': {'unit': '$\\mu g/m^3$', 'name': 'PM$_{2.5}$', 'standard': 50},
        }
        self.unit = pollutant_info[pollutant]['unit']
        self.name = pollutant_info[pollutant]['name']
        self.standard = pollutant_info[pollutant].get('standard', None)
#%%
class DONGNAM:
    def __init__(self, city):
        city_info = {
            '경산': {'city':'Gyeongsan', 'province':'Gyeongbuk'},
            '경주': {'city':'Gyeongju', 'province':'Gyeongbuk'},
            '고성': {'city':'Goseong', 'province':'Gyeongnam'},
            '구미': {'city':'Gumi', 'province':'Gyeongbuk'},
            '김해': {'city':'Gimhae', 'province':'Gyeongnam'},
            '대구': {'city':'Daegu', 'province':'Daegu'},
            '부산': {'city':'Busan', 'province':'Busan'},
            '양산': {'city':'Yangsan', 'province':'Gyeongnam'},
            '영천': {'city':'Yeongcheon', 'province':'Gyeongbuk'},
            '울산': {'city':'Ulsan', 'province':'Ulsan'},
            '진주': {'city':'Jinju', 'province':'Gyeongnam'},
            '창원': {'city':'Changwon', 'province':'Gyeongnam'},
            '칠곡': {'city':'Chilgok', 'province':'Gyeongbuk'},
            '포항': {'city':'Pohang', 'province':'Gyeongbuk'},
            '하동': {'city':'Hadong', 'province':'Gyeongnam'}
        }
        self.city = city_info[city]['city']
        self.province = city_info[city]['province']
#%%
class PROVINCE:
    def __init__(self, province):
        province_info = {
            '경북': 'Gyeongbuk',
            '경남': 'Gyeongnam',
            '부산': 'Busan',
            '울산': 'Ulsan',
            '대구': 'Daegu'
        }
        self.province = province_info[province]
        
    def __str__(self):
        return self.province