# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:20:48 2021

@author: Ruben
"""

import rasterio
from rasterio.plot import show 
import georaster
import matplotlib.pyplot as plt

pathfile = r'C://Users/Ruben/Documents/Thesis/Data/Imagetest/'

fp = pathfile + '20201228_033335_48_2424_3B_AnalyticMS_clip.tiff'

img = georaster.MultiBandRaster(fp)


