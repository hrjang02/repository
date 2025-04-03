import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import math
import netCDF4
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair)
from datetime import datetime
from datetime import timedelta
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import interpolate
import glob

bottom = 0; top = 75;
tt=0; count=0
inum = 200; jnum= 200;

time = ['15','16','17','18','19','20','21','22','23']

for i in time:

    wrf_file= Dataset('/data02/erkim/LES/US/wrfout_d04_2022-09-29_'+i)

    PBLHt = np.zeros((jnum,inum))
    print(i)
    
    for jj in range(0,jnum):
        for ii in range(0,inum):

            ht = getvar(wrf_file, "height_agl", units='m', timeidx=tt)
            ht = ht[0:75,jj,ii]
            ter = getvar(wrf_file, "ter", timeidx=tt)
            ter = ter[jj,ii]

            U = wrf_file.variables['U'][:,bottom:top,jj,ii] #(60, 50, 200, 201)
            V = wrf_file.variables['V'][:,bottom:top,jj,ii] 
            W = wrf_file.variables['W'][:,bottom:top,jj,ii]
            GRD = wrf_file.variables['GRDFLX'][:,:,:]
            Stb = GRD.mean()

            POT = wrf_file.variables['T'][:,bottom:top,jj,ii]+300 # K #(60, 50, 200, 200)
            QV = wrf_file.variables['QVAPOR'][:,bottom:top,jj,ii] #Water vapor mixing ratio(kg/kg) -> g/kg
            POTV = POT*(1+0.61*QV)
            #print(POTV.shape) #(60, 50, 200, 200)

            u_ave = U.mean(axis=0) ; u_var = U.var(axis=0)
            v_ave = V.mean(axis=0) ; v_var = V.var(axis=0)
            w_ave = W.mean(axis=0) ; w_var = W.var(axis=0)
            pot_ave = POT.mean(axis=0) ; pot_var = POT.var(axis=0)
            potv_ave = POTV.mean(axis=0) ; potv_var = POTV.var(axis=0) #(50, 200, 200)

            TKE = 0.5*(u_var + v_var + w_var)

            u = u_ave[:]; v = v_ave[:]
            potv = potv_ave[:] 
        

            alt_new = np.arange(50, 2550, 50)
            U_new = np.empty((50,1))
            V_new = np.empty((50,1))
            POT_new = np.empty((50,1))
            POTV_new = np.empty((50,1))
            TKE_new = np.empty((50,1))
            #print(alt_new)

            alt_ori = ht
            u_ori = u 
            v_ori = v
            potv_ori = potv
            tke_ori = TKE

            func = interpolate.interp1d(alt_ori, potv_ori, bounds_error=False)
            potv_intp = func(alt_new)
            POTV_new = np.column_stack((POTV_new, potv_intp))
            POTV_new = np.delete(POTV_new,0,1)

            func = interpolate.interp1d(alt_ori, u_ori, bounds_error=False)
            u_intp = func(alt_new)
            U_new = np.column_stack((U_new, u_intp))
            U_new = np.delete(U_new,0,1)

            func = interpolate.interp1d(alt_ori, v_ori, bounds_error=False)
            v_intp = func(alt_new)
            V_new = np.column_stack((V_new, v_intp))
            V_new = np.delete(V_new,0,1)

            func = interpolate.interp1d(alt_ori, tke_ori, bounds_error=False)
            tke_intp = func(alt_new)
            TKE_new = np.column_stack((TKE_new, tke_intp))
            TKE_new = np.delete(TKE_new,0,1)


            dummy_u = np.hstack([U_new])
            dummy_v = np.hstack([V_new])
            dummy_potv = np.hstack([POTV_new])
            dummy_tke = np.hstack([TKE_new])
            #print(dummy_u.shape)

            save_u = dummy_u.copy()
            save_v = dummy_v.copy()
            save_potv = dummy_potv.copy()
            save_tke = dummy_tke.copy()
            #print(save_tke)

            Rib = save_u.copy()

            for ll in range(1,50):

                z = alt_new[ll]
                zs = alt_new[0]
                z = int(z)
                zs = int(zs)

                potvs = save_potv[0]
                us = save_u[0]
                vs = save_v[0]

                Rib[ll] = (9.81/potvs)*(((save_potv[ll]-potvs)*(z-zs)) / \
                ( ((save_u[ll]-us)**2) * ((save_v[ll]-vs)**2)) )
        #    print(Rib.shape)


            if Stb < 0: 
                for ll in range (1,50):
                    if Rib[ll] > 0.3:
                        #print('Rib',ll,Rib[ll]) 
                        PBLH = ll*50+50
                        break
                    else:
                        PBLH = 50
            else:

                for ll in range (1,50):
                    if save_tke[ll] > 0.2:
                        #print('tke',ll,save_tke[ll])
                        PBLH = ll*50+50
                        break
                    else:
                        PBLH = 50


            PBLHt[jj,ii] = PBLH


    ncfile= '/data02/erkim/LES/US/wrfout_d04_2022-09-29_'+i
    data = netCDF4.Dataset(ncfile, 'r+')

    for j in range(0,jnum):
        for i in range(0,inum):

            data['PBLH'][:,j,i] = PBLHt[j,i]

    data.close()
    print('Finished.')

