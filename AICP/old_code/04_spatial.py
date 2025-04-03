import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon, Point
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib 
from cartopy import geodesic as cgeo
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

def korea_map(
    extent,
    grid_lon_level,
    grid_lat_level,
    grid_bool=True,
    res="m",
    road=False,
    crs=ccrs.PlateCarree(),
    road_region="Ulsan",
    figsize=(5, 5),
):

    if res == "h":
        shp_fname = "/data02/Map_shp/TL_SCCO_SIG_WGS_cp949.shp"
    if res == "m":
        shp_fname = "/data02/Map_shp/TL_SCCO_CTPRVN_WGS_utf8.shp"
    reader = Reader(shp_fname)
    geometry = reader.geometries()
    kor_shape = ShapelyFeature(
        geometry, ccrs.PlateCarree(), edgecolor="k", facecolor="None"
    )

    fig, ax = plt.subplots(
        subplot_kw=dict(projection=crs), facecolor=(1, 1, 1), figsize=figsize
    )
    ax.set_extent(extent)

    if road:
        link = pd.read_csv(
            f"/data02/Nodelink_data/20220905/WGS84_converted/{road_region}_link_20220905.csv"
        )
        link["geometry"] = link["geometry"].apply(wkt.loads)
        link_gpd = gpd.GeoDataFrame(link, crs="epsg:4326")
        ax.add_geometries(
            link_gpd["geometry"],
            crs=crs,
            edgecolor="k",
            facecolor="None",
            linewidth=0.2,
            alpha=0.5,
        )

    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = matplotlib.ticker.FixedLocator(
        np.arange(
            extent[0] - grid_lon_level * 2,
            extent[1] + grid_lon_level * 2,
            grid_lon_level,
        )
    )  # Set Xtick labels
    gl.ylocator = matplotlib.ticker.FixedLocator(
        np.arange(
            extent[2] - grid_lat_level * 2,
            extent[3] + grid_lat_level * 2,
            grid_lat_level,
        )
    )

    if not grid_bool:
        gl.xlines = False
        gl.ylines = False

    ax.add_feature(kor_shape, linewidth=0.3)

    return fig, ax



file_list = ['0930']
for i in range(0,np.shape(file_list)[0]):
    data_path = '/home/erkim/03_WRFLES/NO2_v4/IUP_Ulsan_'+file_list[i]+'.csv'
    data = np.loadtxt(data_path, delimiter=',', skiprows = 1, usecols=(2,3,4,5))

    lon = data[:,0]
    lat = data[:,1]

    NO2 = data[:,3]




lon_min = 129.29
lon_max = 129.40#np.max(lon)
lat_min = 35.38#np.min(lat)
lat_max = 35.50#np.max(lat)

#df = pd.read_csv()
fig, ax = korea_map([lon_min, lon_max, lat_min, lat_max], 0.02, 0.02, road=True)
ax.scatter(lon, lat, c=NO2, cmap='jet')
#ax.scatter(lon, lat, c=k, cmap='jet', projection=ccrs.PlateCarree())
plt.show()
