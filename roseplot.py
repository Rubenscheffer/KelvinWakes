# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:04:49 2021

@author: Ruben
"""

from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

datapath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_complete.csv'

df = pd.read_csv(datapath)

#%%Wind

ws = df['ones']
wd = df['wind_angle']


  # 
ax = WindroseAxes.from_ax()
ax.set_xticklabels(['E', 'NE',  'N', 'NW', 'W', 'SW','S', 'SE'])
# ax.set_theta_zero_location('N')
ax.bar(wd, ws, normed=True, opening=0.4,nsector=36, edgecolor='black')
# ax.set_legend(loc=((1.1,0.5)))

#%% Current

cs = df['ones']
cd = df['current_heading']

bx = WindroseAxes.from_ax()
bx.set_xticklabels(['E', 'NE',  'N', 'NW', 'W', 'SW','S', 'SE'])
# bx.set_theta_zero_location('N')
bx.bar(cd, cs, normed=True, opening=0.4,nsector=36, edgecolor='black')
# bx.set_legend(loc=((1.1,0.5)))

#%% Ship direction

ss = df['ones']
sd = df['Turb_heading']

cx = WindroseAxes.from_ax()
cx.set_xticklabels(['E', 'NE',  'N', 'NW', 'W', 'SW','S', 'SE'])
# cx.set_theta_zero_location('N')
cx.bar(sd, ss, normed=True, opening=0.4, nsector=36, color='blue', edgecolor='black')
# bx.set_legend(loc=((1.1,0.5)))