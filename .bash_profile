
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

############### Intel Compiler Setting Part ###############
#@@@ Intel Parallel Studio XE 2018 update3 Cluster Edition (compiler v18)
source /usr/local/intel/bin/compilervars.sh intel64    	## Intel v18(Inc.Intel-MPI)
#source /usr/local/mpi/intel18/mvapich2-2.3.4/bin/mpivars.sh    ## MVAPICH2-2.3.4
#source /usr/local/mpi/intel18/openmpi-3.1.6/bin/mpivars.sh     ## OPENMPI-3.1.6
#export HDF5=/usr/local/hdf5/1.10.5_intel18           		## HDF5-1.10.5
#export NETCDF=/usr/local/netcdf/3.6.3_intel18        		## NetCDF-3.6.3
#export NETCDF=/usr/local/netcdf/4.0.1_intel18_no_nc4     	## NetCDF-4.0.1_No_NC4
#export NETCDF=/usr/local/netcdf/4.0.1_intel18            	## NetCDF-4.0.1
#export NETCDF=/usr/local/netcdf/4.1.3_intel18            	## NetCDF-4.1.3
export NETCDF=/usr/local/netcdf/4.6.1_intel18            	## NetCDF-4.6.1 with NetCDF-Fortran-4.4.5
#export PNETCDF=/usr/local/pnetcdf/1.11.2_intel18_mvapich2-2.3.4	## Parallel-NetCDF-1.11.2 with mvapich2-2.3.4
#export PIO=/usr/local/pio/2.3.1_intel18_mvapich2-2.3.4		## Parallel IO-2.3.1 with mvapich2-2.3.4
###########################################################

############## NVIDIA Compiler Setting Part ################
#@@@ NVIDIA HPC SDK (include PGI Compiler & CUDA11 & MATH LIB & OpenMPI)
#source /usr/local/nvidia/Linux_x86_64/20.7/compilers/bin/nv.sh ## NV(PGI) v20.7
#source /usr/local/mpi/nv20/mvapich2-2.3.4/bin/mpivars.sh      	## MVAPICH2-2.3.4
#source /usr/local/mpi/nv20/openmpi-3.1.6/bin/mpivars.sh     	## OPENMPI-3.1.6
#export HDF5=/usr/local/hdf5/1.10.5_nv20                        ## HDF5-1.10.5
#export NETCDF=/usr/local/netcdf/3.6.3_nv20                     ## NetCDF-3.6.3
#export NETCDF=/usr/local/netcdf/4.0.1_nv20_no_nc4              ## NetCDF-4.0.1_No_NC4
#export NETCDF=/usr/local/netcdf/4.0.1_nv20                     ## NetCDF-4.0.1
#export NETCDF=/usr/local/netcdf/4.1.3_nv20                     ## NetCDF-4.1.3
#export NETCDF=/usr/local/netcdf/4.6.1_nv20                     ## NetCDF-4.6.1 with NetCDF-Fortran-4.4.5
#export PNETCDF=/usr/local/pnetcdf/1.11.2_nv20_mvapich2-2.3.4   ## Parallel-NetCDF-1.11.2 with mvapich2-2.3.4
#export PIO=/usr/local/pio/2.3.1_nv20_mvapich2-2.3.4            ## Parallel IO-2.3.1 with mvapich2-2.3.4
#############################################################

############### The Others Setting Part ###################
#export NCARG_ROOT=/usr/local/ncl_ncarg/662/gcc485   		## Ncl_ncarg-6.6.2 with gcc-4.8.5

#export GrADS=/usr/local/grads/2.1.0.oga.1      		## OpenGrads-2.1.0.oga.1
#export GADDIR=/usr/local/grads/2.1.0.oga.1/data         	## OpenGrads-2.1.0.oga.1 Fonts, etc.

#export HDF4=/usr/local/hdf4/4.2.10                   		## HDF-4.2.10
export HDF5=/usr/local/hdf5/1.10.5                  		## HDF5-1.10.5

#export NETCDF=/usr/local/netcdf/4.6.1_gcc485	     		## NetCDF-4.6.1 with gcc-4.8.5
export NCVIEW=/usr/local/ncview/2.1.4		     		## Ncvivew-2.1.4
export NCO=/usr/local/nco/4.8.0		   		## NCO-4.8.0
#export HDFVIEW=/usr/local/HDFView/2.10.1                	## HDFView-2.10.1
#export GMT=/usr/local/gmt/6.0.0                        	## GMT-6.0.0
#source /usr/local/ferret/6.93/ferret_paths.sh          	## ferret-6.93
#export CDO=/usr/local/cdo/1.9.3                        	## CDO-1.9.3

export JASPER=/usr/local/jasper/1.900.1			## jasper-1.900.1
export JASPERINC=$JASPER/include				## jasper include PATH
export JASPERLIB=$JASPER/lib					## jasper library PATH
###########################################################

export PATH=$PATH:$HOME/bin:.:~:
export MANPATH=$MANPATH
export LD_LIBRARY_PATH=/home/hrjang2/anaconda3/lib:$LD_LIBRARY_PATH
for i in $NETCDF $PNETCDF $PIO $NCARG_ROOT $GrADS $HDF4 $HDF5 $NCO $NCVIEW $HDFVIEW $GMT $CDO ; do
  if [ ! -z $i ]; then
    if [ -d $i/bin ]; then export PATH=$i/bin:$PATH ; fi
    if [ -d $i/lib ]; then export LD_LIBRARY_PATH=$i/lib:$LD_LIBRARY_PATH ; fi
    if [ -d $i/sbin -a `id -u` = 0 ]; then export PATH=$i/sbin:$PATH ; fi
    if [ -d $i/man ]; then export MANPATH=$MANPATH:$i/man ; fi
  fi
done
unset i
