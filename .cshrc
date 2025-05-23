# *****************************************************************
# *                 CentOS tcsh Script                            *
# *****************************************************************

########### Intel Cluster Toolkit Setting Part ############
#@@@ Intel Parallel Studio XE 2018 update3 Cluster Edition (compiler v18)
source /usr/local/intel/bin/compilervars.csh intel64    	 ## Intel v18(Inc.Intel-MPI)
source /usr/local/mpi/intel18/mvapich2-2.3.4/bin/mpivars.csh    ## MVAPICH2-2.3.4
#source /usr/local/mpi/intel18/openmpi-3.1.6/bin/mpivars.csh     ## OPENMPI-3.1.6
#setenv HDF5 /usr/local/hdf5/1.10.5_intel18            		 ## HDF5-1.10.5
#setenv NETCDF /usr/local/netcdf/3.6.3_intel18         		 ## NetCDF-3.6.3
#setenv NETCDF /usr/local/netcdf/4.0.1_intel18_no_nc4         	 ## NetCDF-4.0.1_NO_NC4
#setenv NETCDF /usr/local/netcdf/4.0.1_intel18         		 ## NetCDF-4.0.1
#setenv NETCDF /usr/local/netcdf/4.1.3_intel18         		 ## NetCDF-4.1.3
setenv NETCDF /usr/local/netcdf/4.6.1_intel18         		 ## NetCDF-4.6.1 with NetCDF-Fortran-4.4.5
#setenv PNETCDF /usr/local/pnetcdf/1.11.2_intel18_mvapich2-2.3.4 ## Parallel-NetCDF-1.11.2 with mvapich2-2.3.4
setenv PIO /usr/local/pio/2.3.1_intel18_mvapich2-2.3.4          ## Parallel IO-2.3.1 with mvapich2-2.3.4
###########################################################

############# NVIDIA Compiler Setting Part ################
#@@@ NVIDIA HPC SDK (include PGI Compiler & CUDA11 & MATH LIB & OpenMPI)
source /usr/local/nvidia/Linux_x86_64/20.7/compilers/bin/nv.csh ## NV(PGI) v20.7
#source /usr/local/mpi/nv20/mvapich2-2.3.4/bin/mpivars.csh     	 ## MVAPICH2-2.3.4
#source /usr/local/mpi/nv20/openmpi-3.1.6/bin/mpivars.csh    	 ## OPENMPI-3.1.6
#setenv HDF5 /usr/local/hdf5/1.10.5_nv20                         ## HDF5-1.10.5
#setenv NETCDF /usr/local/netcdf/3.6.3_nv20                      ## NetCDF-3.6.3
#setenv NETCDF /usr/local/netcdf/4.0.1_nv20_no_nc4               ## NetCDF-4.0.1_No_NC4
#setenv NETCDF /usr/local/netcdf/4.0.1_nv20                      ## NetCDF-4.0.1
#setenv NETCDF /usr/local/netcdf/4.1.3_nv20                      ## NetCDF-4.1.3
#setenv NETCDF /usr/local/netcdf/4.6.1_nv20                      ## NetCDF-4.6.1 with NetCDF-Fortran-4.4.5
#setenv PNETCDF /usr/local/pnetcdf/1.11.2_nv20_mvapich2-2.3.4    ## Parallel-NetCDF-1.11.2 with mvapich2-2.3.4
#setenv PIO /usr/local/pio/2.3.1_nv20_mvapich2-2.3.4             ## Parallel IO-2.3.1 with mvapich2-2.3.4
###########################################################

############### The Others Setting Part ###################
setenv NCARG_ROOT /usr/local/ncl_ncarg/662/gcc485      	## Ncl_ncarg-6.6.2 with gcc-4.8.5

setenv GrADS /usr/local/grads/2.1.0.oga.1            		## OpenGrads-2.1.0.oga.1
setenv GADDIR /usr/local/grads/2.1.0.oga.1/data        	## OpenGrads-2.1.0.oga.1 Fonts, etc.

setenv HDF4 /usr/local/hdf4/4.2.10                   		## HDF-4.2.10
setenv HDF5 /usr/local/hdf5/1.10.5                   		## HDF5-1.10.5

#setenv NETCDF /usr/local/netcdf/4.6.1_gcc485			## NetCDF-4.6.1 with gcc-4.8.5
setenv NCVIEW /usr/local/ncview/2.1.4				## NCView-2.1.4
setenv NCO /usr/local/nco/4.8.0				## NCO-4.8.0
setenv HDFVIEW /usr/local/HDFView/2.10.1			## HDFView-2.10.1
setenv GMT /usr/local/gmt/6.0.0                        	## GMT-6.0.0
source /usr/local/ferret/6.93/ferret_paths.csh          	## ferret-6.93
setenv CDO /usr/local/cdo/1.9.3                        	## CDO-1.9.3

setenv JASPER /usr/local/jasper/1.900.1                        ## jasper-1.900.1
setenv JASPERINC ${JASPER}/include                             ## jasper include PATH
setenv JASPERLIB ${JASPER}/lib                                 ## jasper library PATH
###########################################################
##### Two-Way WRF-CMAQ #####
setenv WRF_CMAQ 1
setenv IOAPI /usr/local/ioapi/3.2 
setenv WRFIO_NCD_LARGE_FILE_SUPPORT 1

##### SMOKE HOME #####
setenv SMK_HOME /home/yscha/SMOKE/v450/SMOKE_UNIST

alias 01 'ssh greenworld01'
alias 02 'ssh greenworld02'
alias 03 'ssh greenworld03'
alias 04 'ssh greenworld04'
alias 05 'ssh greenworld05'
alias 06 'ssh greenworld06'
alias 07 'ssh greenworld07'
alias 08 'ssh greenworld08'

#foreach i (${NETCDF} ${PNETCDF} ${PIO} ${NCARG_ROOT} ${GrADS} ${HDF4} ${HDF5} ${NCO} ${NCVIEW} ${HDFVIEW} ${GMT} ${CDO})
foreach i (${NETCDF} ${NCARG_ROOT} ${GrADS} ${HDF5} ${NCO} ${NCVIEW} ${HDFVIEW} ${GMT} ${CDO})
  if (! -z ${i}) then ;
    if ( -d ${i}/bin) then ; setenv PATH ${i}/bin:${PATH} ; endif
    if ( -d ${i}/lib) then ; setenv LD_LIBRARY_PATH ${i}/lib:${LD_LIBRARY_PATH} ; endif
    if ( -d ${i}/sbin && `id -u` == 0) then ; setenv PATH ${i}/sbin:${PATH} ; endif
    if ( -d ${i}/man) then ; setenv MANPATH ${MANPATH}:${i}/man ; endif
  endif
end
unset i

alias hdfview "/usr/local/HDFView/2.10.1/bin/hdfview.sh"
alias package "cd /home/hrjang2/anaconda3/envs/hyoran/lib/python3.9/site-packages"

#setenv LD_LIBRARY_PATH /usr/local/hdf5/1.10.5/lib:/home/smkim2/anaconda3/lib:$LD_LIBRARY_PATH
setenv LD_LIBRARY_PATH /usr/local/hdf5/1.10.5/lib:$LD_LIBRARY_PATH


# >>> conda initialize >>>
 # !! Contents within this block are managed by 'conda init' !!
if ( -f "/home/hrjang2/anaconda3/etc/profile.d/conda.csh" ) then
    source "/home/hrjang2/anaconda3/etc/profile.d/conda.csh"
else
    setenv PATH "/home/hrjang2/anaconda3/bin:$PATH"
endif
# <<< conda initialize <<<

