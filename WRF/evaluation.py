#%%
import netCDF4 as nc
#%%
wrfout_path = '/home/hrjang2/2_LES/WRF/run/wrfout_d03_2022-09-01_00:00:00'
wrfout_file = nc.Dataset(wrfout_path, 'r')

print(wrfout_file.variables.keys())
print(len(wrfout_file.variables.keys()))
# %%
