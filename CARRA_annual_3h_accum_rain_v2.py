#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:52:21 2021

@author: jeb
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 
  
wo=1
do_plt=0

# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'
max_value=6000

def loadct():
    r=[188,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156, 255, 255]
    g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28, 255, 255 ]
    b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196, 0, 255 ]
    r=[255,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156]
    g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28 ]
    b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196]
    colors = np.array([r, g, b]).T / 255
    n_bin = 24
    cmap_name = 'my_list'
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    return cm

# for a later version that maps the result
ni=1269
nj=1069

working_remote=0

inpath='/Users/jason/0_dat/CARRA/'

outpath='/Users/jason/0_dat/CARRA/output/'
if working_remote:
    inpath='/Users/jason/0_dat/CARRA/'
    outpath='/Users/jason/0_dat/CARRA/output/'

# ------------------------------------------- rain fraction
rainos=0.
x0=0.5 ; x1=2.5
x0-=rainos
x1-=rainos
y0=0 ; y1=1
a1=(y1-y0)/(x1-x0)
a0=y0-a1*x0
               
map_version=1
  
AW=0
path='/Users/jason/Dropbox/CARRA/prog/map_CARRA_west/'
if AW:path='/Users/jason/Dropbox/CARRA/prog/map_CARRA_west/'
os.chdir(path)

if map_version:
    fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
    lat=np.fromfile(fn, dtype=np.float32)
    lat=lat.reshape(ni, nj)

    fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
    lon=np.fromfile(fn, dtype=np.float32)
    lon=lon.reshape(ni, nj)

    # latx=np.rot90(lat.T)
    # lonx=np.rot90(lon.T)
    offset=0
    lon=lon[offset:ni-offset,offset:nj-offset]
    lat=lat[offset:ni-offset,offset:nj-offset]
    ni-=offset*2
    nj-=offset*2
    # print(ni,nj)
    LLlat=lat[0,0]
    LLlon=lon[0,0]-360
    # print("LL",LLlat,LLlon)
    # print("UL",lat[ni-1,0],lon[ni-1,0]-360)
    lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
    lat0=lat[int(round(ni/2)),int(round(nj/2))]
    # print("mid lat lon",lat0,lon0)
    
    URlat=lat[ni-1,nj-1]
    URlon=lon[ni-1,nj-1]
    # print("LR",lat[0,nj-1],lon[0,nj-1]-360)
    # print("UR",URlat,URlon)

    # m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=72, lon_0=-36, resolution='l', projection='lcc')
    m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=lat0, lon_0=lon0, resolution='l', projection='lcc')
    x, y = m(lat, lon)


year='2019'
ts=pd.Timestamp(int(year),1,1) ; day0=ts.dayofyear
ts=pd.Timestamp(int(year),12,31) ; day1=ts.dayofyear

# year='2017'
# ts=pd.Timestamp(int(year),9,1) ; day0=ts.dayofyear
# ts=pd.Timestamp(int(year),9,30) ; day1=ts.dayofyear

files = sorted(glob(inpath+'CARRA-West_prec_*'+year+'*'+'3h*.nc'))
print(files)

print(day0,day1)

# day0s=str(day0) ; day0s.rjust(3 + len(day0s), '0') 
# res = datetime.strptime(year + "-" + str(day0), "%Y-%j").strftime("%m-%d-%Y") 
# my_date = datetime.strptime(res, "%m-%d-%Y")

# print(my_date)
# nc_t2m = Dataset(inpath+'CARRA-West_T2m_mean_'+year+'.nc', mode='r')
# print(nc_t2m.variables)

n_days=day1-day0+1

choice=1

for choice in range(0,3):

# for choice in range(1,2):
    datacube=np.zeros(((n_days,ni,nj)))
    if choice==0:
        varnam='t2m'
        varnam2=r'$(t2m_{max} + t2m_{min})/2$'
        cm='jet'
        units='°C'
        extra=''
    if choice==1:
        varnam='rf'
        varnam2='rainfall'
        extra=', phase transition '+str(x0)+' to '+str(x1)+'°C after Hock and Holmgren (2005)'
        units='mm / 3 h'
    if choice==2:
        varnam='tp'
        varnam2='total precip.'
        extra=''
        units='mm / 3 h'
    for day_index in range(day0,day1):
    # for day_index in range(day0,day0+1):
        day0s=str(day_index) ; day0s.rjust(3 + len(day0s), '0') 
        res = datetime.strptime(year + "-" + str(day_index), "%Y-%j").strftime("%m-%d-%Y") 
        my_date = datetime.strptime(res, "%m-%d-%Y")
        print(varnam,day0,day1,day_index,my_date)
        # print(datetime.strptime(res, "%m-%d-%Y"))
        # for i,fn in enumerate(files[0:1]):
        for i,fn in enumerate(files):
            plt.close()
            # print(fn,i)
            h_offset=int(fn[53:55])
            h0=int(fn[63:65])+h_offset
            h1=int(fn[66:68])+h_offset
            
            nc = Dataset(fn, mode='r')
            # print(i,nc,nc.variables)
            # j0=700 ; j1=1210
            # i0=180 ; i1=580
            # j0=900 ; j1=1210
            # i0=180 ; i1=i0+200
    
            # tp = nc.variables['TPRATE_surface'][day_index-1,j0:j1,i0:i1]
            tp = nc.variables['TPRATE_surface'][day_index-1,:,:]
            nam='mean_air_temperature_2m'
            if int(year)>2017:nam='TMP_2maboveground'
            t2m=nc_t2m.variables[nam][(day_index-1)*12+i+4,:,:]-273.15
            if int(year)==2018:
                t2m=np.rot90(t2m.T)

            f=np.zeros((ni,nj))
            v=np.where(((t2m>x0)&(t2m<x1)))
            f[v]=t2m[v]*a1+a0
            v=np.where(t2m>x1) ; f[v]=1
            v=np.where(t2m<x0) ; f[v]=0
            # plt.close()
            # plt.imshow(np.rot90(t2m.T))
            # plt.show()
#%%
            rf=tp*np.rot90(f.T)
            # print(i,np.sum(datacube[day_index-1,:,:]))
            
            if choice==0:
                plotvar=t2m
            if choice==1:
                plotvar=np.rot90(rf.T)
            if choice==2:
                plotvar=np.rot90(tp.T)
    
            datacube[day_index-1,:,:]+=plotvar
    
                # plotvar=np.rot90(plotvar.T)
    #         A mixture of rain and snow is assumed for a transition zone ranging from 1 K above and 1 K below
    # the threshold temperature. Within this temperature range,
    # the snow and rain percentages of total precipitation are
    # obtained from linear interpolation.
            # t2m
            # plt.close()
            # plt.imshow(rf)
            # plt.show()
          #  #%%
            # sum_tp=int(np.sum(rf))
            # # if np.sum(tp)>0:
            # print(sum_tp)
            # if sum_tp>0:
            if do_plt:
            # if sum_tp>9e4:
                # print("making graphic",tp.shape,np.sum(tp))
                # make graphic
                ax = plt.subplot(111)
                dates=year+'-'+f'{my_date.month:02d}'+'-'+f'{my_date.day:02d}'+'-'+f'{h0:02d}'+'-'+f'{h1:02d}'
                dates2=year+'-'+f'{my_date.month:02d}'+'-'+f'{my_date.day:02d}'+' '+f'{h0:02d}'+'-'+f'{h1:02d}h'
                tit='CARRA '+varnam2+' '+dates2
                ax.set_title(tit)
                max_value=12
                if choice>0:cm=loadct()
                if map_version==0:
                    pp=plt.imshow(plotvar, interpolation='nearest', origin='lower', cmap=cm,vmin=0,vmax=max_value) ; plt.axis('off') 
                
                if map_version:
                    pp=m.imshow(plotvar, cmap = cm,vmin=0,vmax=max_value) 
                    # m.axis('off')
                    m.drawcoastlines(color='k',linewidth=0.5)
                    m.drawparallels([66.6,83.65],color='gray')
                    m.drawparallels([60,70,80,83.65],dashes=[2,4],color='k')
                    m.drawmeridians(np.arange(0.,420.,10.))
                    # m.drawmapboundary(fill_color='aqua')
                    ax = plt.gca()     
                    # plt.title("Lambert Conformal Projection")
                    # plt.show()
    
                if choice>0:
                    cbar_min=0
                    cbar_max=max_value
                    cbar_step=2
                    cbar_num_format = "%d"
        
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    
                    # plt.colorbar(im)            
                    cbar = plt.colorbar(pp,
                                        orientation='vertical',
                                        ticks=np.arange(cbar_min,
                                        cbar_max+cbar_step, cbar_step),format=cbar_num_format, cax=cax)
                    cbar.ax.set_ylabel(units, fontsize = font_size)
                    tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
                    # print(tickranges)
                    cbar.ax.set_yticklabels(tickranges, fontsize=font_size)
    
                if choice==0:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    
                    # plt.colorbar(im)            
                    cbar = plt.colorbar(pp,
                                        orientation='vertical',format=cbar_num_format, cax=cax)
                    cbar.ax.set_ylabel(units, fontsize = font_size)
                    # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
                    # # print(tickranges)
                    # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)
    
                cc=0
                xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
                mult=0.7
                color_code='grey'
                plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team'+extra, fontsize=font_size*mult,
                  transform=ax.transAxes,color=color_code) ; cc+=1. 
                
                ly='p'
                if ly == 'x':
                    plt.show()
                DPI=100
                # DPI=300
                
                if ly == 'p':
                    figpath='/Users/jason/0_dat/CARRA/Fig/3h/'
                    # figname=figpath+str(sum_tp)+dates
                    figname=figpath+dates+'_'+varnam
                    plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
    
                # cxcccx
    
        make_gif=0
        if make_gif:
            animpath='/Users/jason/Dropbox/CARRA/prog/anim/'
            animname=year+'-'+str(month)+'-'+str(day)+'_'+varnam
            # animname='2017_Sep_1-30_'+varnam
            inpath=figpath
            msg='convert  -delay 100  -loop 0   '+inpath+'*.png   '+animpath+animname+'.gif'
            os.system(msg)
    
    if choice==0:datacube[:,:,:]/=8
    
    if wo:
        ofile=outpath+varnam+'_'+year+'.nc'
        datacube
        os.system("/bin/rm "+ofile)
        ncfile = Dataset(ofile,mode='w',format='NETCDF4_CLASSIC')
        lat_dim = ncfile.createDimension('lat', nj)     # latitude axis
        lon_dim = ncfile.createDimension('lon', ni)    # longitude axis
        time_dim = ncfile.createDimension('time', n_days) # unlimited axis (can be appended to)
        
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
        temp.units = 'mm/day' # degrees Kelvin
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
        temp[:,:,:] = datacube  # Appends data along unlimited dimension
    
        # temp2 = ncfile.createVariable("confidence",np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
        # temp2.units = "unitless" # degrees Kelvin
        # temp2.standard_name = "confidence" # this is a CF standard name
        # temp2[:,:,:] = confidence  # Appends data along unlimited dimension
    
        print("-- Wrote data, temp.shape is now ", temp.shape)
        # read data back from variable (by slicing it), print min and max
        # print("-- Min/Max values:", temp[:,:,:].min(), temp[:,:,:].max())
        
        ncfile.close(); print('Dataset is closed!')
 
