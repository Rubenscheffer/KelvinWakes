# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:48:17 2021

@author: Ruben
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Plot Settings 

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 


#Load data
datapath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_complete.csv'
tempdata = r'C://Users/Ruben/Documents/Thesis/Data/Processing/fulldata.csv'

# multiple

#%%
shipdata = pd.read_csv(datapath)
t_data = pd.read_csv(tempdata)

x = [i + 1 for i in range(12)]


#Calculations

sst_means = []

for column in t_data:
    if column[:3] == 'SST':
        values = t_data[column]
        sst_means.append(np.mean(values.values))


angles = shipdata.Turb_kelvin_angle

#%%Plotting

plt.figure()

plt.scatter(x, sst_means)

plt.xlabel('Time in days after April 8th')
plt.ylabel('Daily SST at 11:00 as a mean of ship locations of ERA 5 reanalysis (K)')
plt.grid(True)

plt.show()

plt.figure()

plt.scatter(shipdata.SST, angles)

plt.xlabel('SST (K)')
plt.ylabel('Turbulent Kelvin Angle (degrees)')
plt.grid(True)

plt.show()
