# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:31:05 2021

@author: Ruben
"""

import xarray as xa
import rioxarray as rio

filepath = r'C:\Users\Ruben\Documents\Thesis\Data\SST20april2.nc'

ncfile = xa.open_dataset(filepath)

print(ncfile.indexes)



# u_s = u_s.rio.set_spatial_dims('longitude', 'latitude')
# u_s.rio.set_crs("epsg:4326")



# Save to GeoTiff
# u_s.rio.to_raster(r"C://Users/Ruben/Documents/Thesis/data/test.tif")
