#%%
#공간분포 그래프 (station별로 201904 평균내서 그리기)
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

file_path = '/data01/OBS/AQMS/2019_PM25.csv'
df = pd.read_csv(file_path)
data_info = df.columns[2:] #첫번째행의 위경도station을 가져오기(이때 '측정일시'는 제외해서 가져옴)

#원하는 month 필터링해서 가져오기
date = '2019-04'
filter = df[df.iloc[:,1].astype(str).str.startswith(date)] 
data = filter.iloc[:,2:].mean(axis=0) #각 열마다 평균내기
data = data.tolist() #평균낸 값을 list화하기

# station number, 위경도 list 추출
std = []
lon = []
lat = []

for info in data_info:
    info = str(info)
    station_info = info.split('_')
    station_number = station_info[0]
    longitude = float(station_info[1])  
    latitude = float(station_info[2])   
    std.append(station_number)
    lon.append(longitude)
    lat.append(latitude)


#그림그리기 시작
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())  #지구를 위경도좌표계에 그리겠다
ax.add_feature(cfeature.COASTLINE) #해안선 추가
ax.set_extent([123, 132, 33, 39]) #지도를 그릴 위경도 범위설정(남한)


cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.1, 'navy'), (0.2, 'blue'),\
            (0.3, 'cornflowerblue'), (0.4, 'turquoise'), (0.5, 'yellowgreen'), (0.6, 'yellow'), (0.7, 'orange'),\
            (0.8, 'orangered'), (0.9, 'red'), (1, 'darkred')], N=80)



scatter = ax.scatter(lon, lat, c=data, cmap=cmap, transform=ccrs.PlateCarree()) #산점도 그리겠다
cbar = fig.colorbar(scatter, location='bottom', label='[μm/m$^3$]') #colorbar 추가

gridlines = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gridlines.xlabels_top = False #X축 라벨을 그리드 라인 위쪽에 표시 안함
gridlines.ylabels_right = False  #Y축 라벨을 그리드 라인 오른쪽에 표시 안함


plt.title('Spatial Distribution of PM2.5 in Korea')
plt.tight_layout()
plt.savefig('/home/intern/shjin/Spatial Distribution of PM2.5 in Korea.png')
#%%