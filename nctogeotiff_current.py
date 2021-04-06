# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:31:05 2021

@author: Ruben
"""

import xarray as xa
import rioxarray as rio

filepath = r'C:\Users\Ruben\Documents\Thesis\Data\current.nc'

ncfile = xa.open_dataset(filepath)
print(ncfile.values)
# print(ncfile.indexes)
surface = ncfile.sel(depth=0)
# print(surface.values)
u_s = surface['uo']
v_s = surface['vo']



#Averaging
u_s = u_s.mean(dim='time')
v_s = v_s.mean(dim='time')

u_s = u_s.rio.set_spatial_dims('longitude', 'latitude')
u_s.rio.set_crs("epsg:4326")
v_s = v_s.rio.set_spatial_dims('longitude', 'latitude')
v_s.rio.set_crs("epsg:4326")

# To check in variables
# u_s = u_s.values
# v_s = v_s.values

# Save to GeoTiff
# u_s.rio.to_raster(r"C://Users/Ruben/Documents/Thesis/data/u_surf.tif")
# v_s.rio.to_raster(r"C://Users/Ruben/Documents/Thesis/data/v_surf.tif")
