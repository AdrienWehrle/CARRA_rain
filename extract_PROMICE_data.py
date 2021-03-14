# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

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

# CARRA grid dims
ni = 1269 
nj = 1069

# read lat lon arrays
fn = './ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
lat_mat = lat.reshape(ni, nj)

fn = './ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
lon_mat = lat.reshape(ni, nj) 

# read PROMICE locations
meta = pd.read_csv('./ancil/PROMICE_info_w_header_2017-2018_stats.csv', 
                   delim_whitespace=True)

# %% point extraction

def match(station_point, era_points):
    
    dists = distance.cdist(station_point, era_points)
    
    dist = np.nanmin(dists)
    
    idx = dists.argmin()
    
    station_gdfpoint = Point(station_point[0, 0], station_point[0, 1])
    matching_era_cell = era_gdfpoints.loc[idx]['geometry']
    
    res = gpd.GeoDataFrame({'promice_station': [station_gdfpoint], 
                            'carra_cell': [matching_era_cell],
                            'distance': pd.Series(dist)})
    
    return res, idx

rows, cols = np.meshgrid(np.arange(np.shape(lat_mat)[1]), 
                         np.arange(np.shape(lat_mat)[0]))

carra_positions = pd.DataFrame({'row': rows.ravel(),
                              'col': cols.ravel(),
                              'lon': lon_mat.ravel(),
                              'lat': lat_mat.ravel()})

carra_points = np.vstack((era_positions.lon.ravel(), 
                          era_positions.lat.ravel())).T

carra_gdfpoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(carra_positions.lon, 
                                                               carra_positions.lat))

# %% time series at PROMICE locations

ds = xr.open_dataset('H:/CARRA/rf_2016.nc')

if AW:
    CARRA_path = 'H:/CARRA/'
if not AW:
    CARRA_path = '/Users/jason/0_dat/CARRA/output/'
    
CARRA_files = glob.glob(CARRA_path + 'rf_*.nc')

for CARRA_file in CARRA_files:
    
    ds = xr.open_dataset(CARRA_file)
