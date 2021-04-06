# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:15:33 2021

@author: Ruben
"""


import netCDF4 as nc
import numpy as np
import gdal



filepath = r'C:\Users\Ruben\Documents\Thesis\Data\SST8april1.nc'
dataset = nc.Dataset(filepath)

print(dataset)






# current = dataset['vo']
# print(current)
# current = current[:]

# lat = dataset['latitude'][:]
# lon = dataset['longitude'][:]

# surface_current = current[:,0,:,:]