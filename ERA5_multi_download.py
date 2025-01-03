#%%
import cdsapi
from multiprocessing import Pool, cpu_count
#%%
year = '2024'
month = '01'
month_length = 31
#%%
def download_pressure_levels(date):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'geopotential', 'relative_humidity', 'specific_humidity',
                'temperature', 'u_component_of_wind', 'v_component_of_wind',
                'vertical_velocity',
            ],
            'pressure_level': [
                '50', '70', '100',
                '125', '150', '175',
                '200', '225', '250',
                '300', '350', '400',
                '450', '500', '550',
                '600', '650', '700',
                '750', '775', '800',
                '825', '850', '875',
                '900', '925', '950',
                '975', '1000',
            ],
            'year': year,
            'month': month,
            'day': date,
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'format': 'grib',
        },
        f'/data01/ERA5_model/{year}/{month}/ERA5_pressure_level_{year}{month}{date}.grib')

def download_single_levels(date):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'land_sea_mask', 'mean_sea_level_pressure',
                'sea_ice_cover', 'sea_surface_temperature', 'skin_temperature',
                'snow_depth', 'soil_temperature_level_1', 'soil_temperature_level_2',
                'soil_temperature_level_3', 'soil_temperature_level_4', 'soil_type',
                'surface_pressure', 'total_precipitation', 'volumetric_soil_water_layer_1',
                'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
            ],
            'year': year,
            'month': month,
            'day': date,
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'format': 'grib',
        },
        f'/data01/ERA5_model/{year}/{month}/ERA5_single_level_{year}{month}{date}.grib')

#%%
if __name__ == '__main__':
    dates = [str(i).zfill(2) for i in range(1, month_length+1)]
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        pool.map(download_pressure_levels, dates)
        pool.map(download_single_levels, dates)

# %%
