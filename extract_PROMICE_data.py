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
from scipy.spatial import distance
from shapely.geometry import Point
from datetime import datetime, timedelta

base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'

AW = 1
if AW:
    base_path = 'C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
    + 'CARRA_rain/'

os.chdir(base_path)

# %% 

def lon360_to_lon180(lon360):

    # reduce the angle  
    lon180 =  lon360 % 360 
    
    # force it to be the positive remainder, so that 0 <= angle < 360  
    lon180 = (lon180 + 360) % 360;  
    
    # force into the minimum absolute value residue class, so that -180 < angle <= 180  
    lon180[lon180 > 180] -= 360
    
    return lon180


# CARRA grid dims
ni = 1269 
nj = 1069

# read lat lon arrays
fn = './ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
lat_mat = lat.reshape(ni, nj)

fn = './ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon = np.fromfile(fn, dtype=np.float32)
lon_pn = lon360_to_lon180(lon)
lon_mat = lon_pn.reshape(ni, nj) 

# read PROMICE locations
meta = pd.read_csv('./ancil/PROMICE_info_w_header_2017-2018_stats.csv', 
                   delim_whitespace=True)

# having a column named "name" create conflicts in DataFrame
meta.rename(columns={'name': 'station_name'}, inplace=True)


# %% point extraction

def match(station_point, CARRA_points):
    
    dists = distance.cdist(station_point, CARRA_points)
    
    dist = np.nanmin(dists)
    
    idx = dists.argmin()
    
    station_gdfpoint = Point(station_point[0, 0], station_point[0, 1])
    matching_CARRA_cell = CARRA_gdfpoints.loc[idx]['geometry']
    
    res = gpd.GeoDataFrame({'promice_station': [station_gdfpoint], 
                            'CARRA_cell': [matching_CARRA_cell],
                            'distance': pd.Series(dist)})
    
    return res, idx

rows, cols = np.meshgrid(np.arange(np.shape(lat_mat)[1]), 
                         np.arange(np.shape(lat_mat)[0]))

CARRA_positions = pd.DataFrame({'row': rows.ravel(),
                                'col': cols.ravel(),
                                'lon': lon_mat.ravel(),
                                'lat': lat_mat.ravel()})

CARRA_points = np.vstack((CARRA_positions.lon.ravel(), 
                          CARRA_positions.lat.ravel())).T

CARRA_gdfpoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(CARRA_positions.lon, 
                                                               CARRA_positions.lat))

# %% match PROMICE locations with CARRA cells 

CARRA_rows = []
CARRA_cols = []

CARRA_cells_atPROMICE = {}

for r, station in meta.iterrows():
    
    station_point = np.vstack((station.lon,
                                station.lat)).T

    # get CARRA cell matching station location
    CARRA_matching_cell, idx = match(station_point, CARRA_points)
    CARRA_matching_rowcol = CARRA_positions.iloc[idx]
    
    CARRA_cells_atPROMICE[station.station_name] = CARRA_matching_rowcol
    
    
# %% time series at PROMICE locations

ds = xr.open_dataset('H:/CARRA/rf_2016.nc')

if AW:
    CARRA_path = 'H:/CARRA/'
if not AW:
    CARRA_path = '/Users/jason/0_dat/CARRA/output/'
    
CARRA_files = glob.glob(CARRA_path + 'rf_*.nc')

for CARRA_file in CARRA_files:
    
    ds = xr.open_dataset(CARRA_file)
    
    year = int(CARRA_file.split(os.sep)[-1].split('.')[0].split('_')[-1])
    
    annual_results = pd.DataFrame()
    
    time = np.arange(datetime(year, 1, 1), datetime(year + 1, 1, 1), 
                     timedelta(days=1)).astype(datetime)
    
    dt_time = pd.to_datetime(time)
    
    annual_results['time'] = dt_time
    
    for r, station in meta.iterrows():
        
        CARRA_location = CARRA_cells_atPROMICE[station.station_name]
        
        # target time series at the point of interest
        CARRA_PROMICE_timeseries = np.array(ds.rf[:,int(CARRA_location.col), 
                                                  int(CARRA_location.row)])
        
        annual_results[station.station_name] = CARRA_PROMICE_timeseries
        
