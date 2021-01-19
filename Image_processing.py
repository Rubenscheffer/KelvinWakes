# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:20:48 2021

@author: Ruben
"""
from osgeo import gdal
import matplotlib.pyplot as plt

pathfile = r'C://Users/Ruben/Documents/Thesis/Data/Imagetest/'

fp = pathfile + '20201228_033335_48_2424_3B_AnalyticMS_clip.tiff'

image = gdal.Open(fp, gdal.GA_ReadOnly)

# Load specific band
band = image.GetRasterBand(4)

arr = band.ReadAsArray()
plt.imshow(arr)
