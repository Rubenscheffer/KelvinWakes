# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:01:43 2021

@author: Ruben
"""


import netCDF4 as nc

filepath = r'C:\Users\Ruben\Documents\Thesis\Data\SST\AQUA_MODIS.20200409T055001.L2.SST.nc'

ncfile = nc.Dataset(filepath)

print(ncfile.variables)