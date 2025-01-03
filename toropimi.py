#%%
import numpy as np
from netCDF4 import Dataset
from glob import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature


# Set Filepath
#sat_path = 'C:/Users/USER/OneDrive - unist.ac.kr/AIR LAB/TROPOMI/'
sat_path = '/data02/SATELLITE/TROPOMI/Products/OFFL/NO2/2019/01/'
file_list = glob(sat_path+'*.nc') #S5P의 모든 파일의 list
file_n = np.size(file_list)  #파일에 담긴 리스트의 크기 반환 = S5P 파일의 갯수

lon_min, lon_max, lat_min, lat_max = extent = [150,110, 20,50]


#for i in range(0,file_n):
for i in range(10,13):
    
    TROPOMI_data = Dataset(file_list[i], mode='r').groups['PRODUCT']  
  
 
    # 차원 줄이는 거
    
    lat = np.squeeze(TROPOMI_data.variables['latitude'][:,:])
    lon = np.squeeze(TROPOMI_data.variables['longitude'][:,:])
    
    NO2_TC = np.squeeze(TROPOMI_data.variables['nitrogendioxide_tropospheric_column'][:,:])*6.02214E19
    NO2_QF = np.squeeze(TROPOMI_data.variables['qa_value'][:,:])


    #NO2_TC[NO2_QF<0.5]=np.nan
     
    split_list = file_list[i].split('_')   #_을 기준으로 list를 만듦
    sensor = 'TROPOMI'                     
    product = split_list[3]                #NO2
   
    date1 = split_list[8]                  
    day1 = date1[0:8]                      #1-8번째 자리까지 가져옴
    hour1 = date1[9:11]                    #9-10번째 자리까지 가져옴
    minute1 = date1[11:13]                 #11-12번째 자리까지 가져옴
    
    date2=split_list[9]
    day2 = date2[0:8]
    hour2 = date2[9:11]
    minute2 = date2[11:13]


    fig = plt.figure(figsize=(5,5))        #지도 사이즈 
    
    ax = plt.axes(projection=ccrs.PlateCarree())     #ccrs.PlateCarree는 원형 지구를 평면으로 표시
    ax.coastlines(resolution='10m', linewidth=0.1)   #해안선 그리기, resolution = 해상도, '10m' = 고해상도, linewidth = 선의 너비
    ax.set_extent([142,110, 20,50])              #지도 확대 범위 

    ax.add_feature(cfeature.COASTLINE)     #해안선 추가     

    
    ax.set_title('%s NO2 %s %s %s:%s - %s %s:%s' %(sensor, product, day1, hour1, minute1, day2,hour2, minute2))
        
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)    #draw는 그리드에 표시, x.y는 표시 안함                             #Draw gridlines
    gl.xlabels_top = False                 #X 축 라벨을 그리드 라인 위쪽에 표시하지 않도록
    gl.ylabels_right = False    
    gl.xlocator = mticker.FixedLocator(np.arange(150,110,10))             #X축 눈금 위치, -180에서 180까지 60 간격으로                                       #Set Xtick labels
    gl.ylocator = mticker.FixedLocator(np.arange(20,50,10))
    
    # smap = ax.scatter(lon,lat,marker="s",s=1,c=NO2_TC,transform=ccrs.PlateCarree())    #lon, lat: 위경도 좌표 데이터, marker="s": 마커 모양을 사각형(square)으로 설정, s=1: 마커의 크기를 1로 설정, c=NO2_TC: 마커의 색깔을 NO2_TC 변수 값에 따라 설정
    smap = ax.pcolormesh(lon, lat, NO2_TC, cmap='jet', shading = 'nearest',transform=ccrs.PlateCarree())    #lon, lat: 위경도 좌표 데이터, marker="s": 마커 모양을 사각형(square)으로 설정, s=1: 마커의 크기를 1로 설정, c=NO2_TC: 마커의 색깔을 NO2_TC 변수 값에 따라 설정
    smap.set_clim([0,1e16])                                                            #산점도의 색깔 범위
    smaplegend = plt.colorbar(smap, location='bottom', pad=0.1)                        #smap: 색깔을 설정한 산점도 객체, location='bottom': 색깔 바의 위치를 아래쪽(bottom)으로 설정, pad=0.1: 색깔 바와 그림 사이의 간격을 설정

    
    smaplegend.set_label('NO2 TCD[molecules/cm$^2$]')
    plt.savefig('/home/hrjang/practice/result/%s NO2 %s %s %s:%s - %s %s:%s.png' %(sensor, product, day1, hour1, minute1, day2,hour2, minute2))
    
    #%%