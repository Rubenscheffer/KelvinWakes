# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:16:27 2021

@author: Ruben
"""


import pickle
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

filepath = r'C:\Users\Ruben\Documents\Thesis\Data\Angles2\Results\sym'

files = os.listdir(filepath)


N = 7

left_angle = np.zeros((N, 5))

right_angle = np.zeros((N, 5))
ship_dir = np.zeros((N, 5))
a_dif = np.zeros((N, 5))

for i, file in enumerate(files):
    filename = filepath + '/' + file

    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
        for j, meas in enumerate(data):
            left_angle[j, i] = meas[1]
            right_angle[j, i] = meas[2]
            a_dif[j, i] = meas[3]
            ship_dir[j, i] = meas[4]
            # print(meas)

#%% Averaging
left_angle_avg = np.mean(left_angle, axis=1)
left_angle_std = np.std(left_angle, axis=1)
right_angle_avg = np.mean(right_angle, axis=1)
right_angle_std = np.std(right_angle, axis=1)
a_dif_avg = np.mean(a_dif, axis=1)
a_dif_std = np.std(a_dif, axis=1)
angle_dif = left_angle_avg - right_angle_avg
ship_dir = np.mean(ship_dir, axis=1)

dat = {'angle_dif': a_dif_avg, 'angle_dif_std': a_dif_std, 'ship_dir': ship_dir}

df = pd.DataFrame(data=dat)






#%% Analysis

# plt.figure()

# plt.errorbar(angle_diffs, a_dif_avg, a_dif_std, linestyle='None', fmt='o', capsize=5)
# plt.hlines(19.47,-190,190, linestyles='dashed', colors='orange')
# plt.grid(True)
# plt.xlabel('Angle between current and travel direction (deg)')
# plt.ylabel('Turbulent Kelvin Angle (deg)')
# plt.tight_layout()
# plt.xlim(-183,183)
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))

# plt.show()