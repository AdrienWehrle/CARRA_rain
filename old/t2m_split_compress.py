#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:18:29 2021

@author: jeb
"""
import os
from netCDF4 import Dataset
import numpy as np

outpath='/Users/jason/0_dat/CARRA/'
d = Dataset(outpath+'CARRA-West_T2m_mean_2016.nc', mode='r')
print(d.variables)

ni=1269
nj=1069

lat=np.zeros(((ni, nj)))
lon=np.zeros(((ni, nj)))
# map_version=1
  
# AW=0
# path='/Users/jason/Dropbox/CARRA/prog/map_CARRA_west/'
# if AW:path='/Users/jason/Dropbox/CARRA/prog/map_CARRA_west/'
# os.chdir(path)

# if map_version:
#     fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
#     lat=np.fromfile(fn, dtype=np.float32)
#     lat=lat.reshape(ni, nj)

#     fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
#     lon=np.fromfile(fn, dtype=np.float32)
#     lon=lon.reshape(ni, nj)
tp = d.variables['mean_air_temperature_2m'][:,:,:]
n_days=tp.shape[0]
print(n_days,n_days/366)

#%%
n_days_per_year=n_days/12

for i in range(12):
    i0=int(i*n_days_per_year)
    i1=int(i0+n_days_per_year)
    print(i,i0,i1,n_days)
    tp = d.variables['mean_air_temperature_2m'][i0:i1,:,:]
    
    varnam='t2m'
    
    ofile=outpath+'CARRA-West_T2m_mean_2016_compressed'+str(i)+'.nc'
    
    os.system("/bin/rm "+ofile)
    ncfile = Dataset(ofile,mode='w',format='NETCDF4_CLASSIC')
    
    
    lat_dim = ncfile.createDimension('lat', nj)     # latitude axis
    lon_dim = ncfile.createDimension('lon', ni)    # longitude axis
    time_dim = ncfile.createDimension('time', n_days_per_year) # unlimited axis (can be appended to)
    
    # for dim in ncfile.dimensions.items():
    #     print(dim)            
    # ncfile.title=varnam+' '+stat_type
    ncfile.subtitle="subtitle"
    # print(ncfile.subtitle)
    # print(ncfile)
    
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'months'
    time.long_name = 'time'
    # Define a 3D variable to hold the data
    print("create with compression")
    temp = ncfile.createVariable(varnam,np.float32,('time','lon','lat'),zlib=True,least_significant_digit=3) # note: unlimited dimension is leftmost
    temp.units = 'K' # degrees Kelvin
    temp.standard_name = varnam # this is a CF standard name
    # print(temp)
    
    nlats = len(lat_dim); nlons = len(lon_dim); ntimes = 3
    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    # lat[:] = -90. + (180./nlats)*np.arange(nlats) # south pole to north pole
    # lon[:] = (180./nlats)*np.arange(nlons) # Greenwich meridian eastward
    # create a 3D array of random numbers
    # data_arr = np.random.uniform(low=280,high=330,size=(ntimes,nlats,nlons))
    # Write the data.  This writes the whole 3D netCDF variable all at once.
    # temp[:,:,:] = np.rot90(result,2)  # Appends data along unlimited dimension
    temp[:,:,:] = tp  # Appends data along unlimited dimension
    
    # temp2 = ncfile.createVariable("confidence",np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
    # temp2.units = "unitless" # degrees Kelvin
    # temp2.standard_name = "confidence" # this is a CF standard name
    # temp2[:,:,:] = confidence  # Appends data along unlimited dimension
    
    print("-- Wrote data, shape is now ", temp.shape)
    # read data back from variable (by slicing it), print min and max
    # print("-- Min/Max values:", temp[:,:,:].min(), temp[:,:,:].max())
    
    ncfile.close(); print('Dataset is closed!')
 