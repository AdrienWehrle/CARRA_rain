#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 09:34:44 2021

@author: jeb
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os
import xarray as xr
import glob
import geopandas as gpd

base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'

AW = 1
if AW:
    base_path = 'C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
    + 'CARRA_rain/'

os.chdir(base_path)

# --------------------------- function for distance on sphere
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2)\
        * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    
    return km

def match(station_point, era_points):
    
    dists = distance.cdist(station_point, era_points)
    
    dist = np.nanmin(dists)
    
    idx = dists.argmin()
    
    station_gdfpoint = Point(station_point[0, 0], station_point[0, 1])
    matching_era_cell = era_gdfpoints.loc[idx]['geometry']
    
    res = gpd.GeoDataFrame({'gistemp_station': [station_gdfpoint], 
                            'era_cell': [matching_era_cell],
                            'distance': pd.Series(dist)})
    
    return res, idx


# CARRA grid dims
ni = 1269 
nj = 1069

# read lat lon arrays
# fn = './ancil/lat_1269x1069.numpy.bin'
fn = './ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
lat = lat.reshape(ni, nj)

# fn = './ancil/lon_1269x1069.numpy.bin'
fn = './ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
lon = lat.reshape(ni, nj) 

# read PROMICE locations
meta = pd.read_csv('./ancil/PROMICE_info_w_header_2017-2018_stats.csv', 
                   delim_whitespace=True)
# print(meta.columns)

n = len(meta)

# locate CARRA grid cells closest to PROMICE AWS
meta["address"] = np.nan

for i in range(n):
    
    dist = haversine_np(meta.lon[i], meta.lat[i], lon, lat)
    minx = np.nanmin(dist)
    temp = dist.flatten()
    v = np.where(temp == minx)
    meta["address"][i] = v[0]
    print(meta.name[i], meta.lon[i], meta.lat[i], minx, v[0])
    
    
# %% read ~500 Mb CARRA infile

if not AW: 
    ds = xr.open_dataset('/Users/jason/0_dat/CARRA/output/rf_2017.nc')

if AW: 
    ds = xr.open_dataset('H:/CARRA/rf_2016.nc')
print(ds)

rf = np.array(ds['rf'])
print(rf.shape)

# %%

# [::-1] flips the array

# test date 220
temp = rf[220,:,:][::-1].flatten()

for i in range(n):
    
    # lat = meta.lat[i]
    # lon = meta.lon[i]
    print(meta.name[i], meta.address[i])
    point = int(meta.address[i])
    # point=[]
    # makes a line around PROMICE stations otherw2ise difficult to see, so high res!
    temp[(int(meta.address[i]) - 20):(int(meta.address[i]) + 20)] = 35
    # print(meta.name[i],meta.lon[i], meta.lat[i],minx,v[0])

temp = temp.reshape(ni, nj)    

plt.imshow(temp)
plt.colorbar()


# %% time series at PROMICE locations

ds = xr.open_dataset('H:/CARRA/rf_2016.nc')

if AW:
    CARRA_path = 'H:/CARRA/'
if not AW:
    CARRA_path = '/Users/jason/0_dat/CARRA/output/'
    
CARRA_files = glob.glob(CARRA_path + 'rf_*.nc')

for CARRA_file in CARRA_files:
    
    ds = xr.open_dataset(CARRA_file)
