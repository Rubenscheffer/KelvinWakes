# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:15:16 2021

@author: Ruben
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% Plot Settings
font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}

matplotlib.rc('font', **font)

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

#%% Main

newpath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_complete.csv'

angledata = pd.read_csv(newpath)


def rotation(x, y):
    ang = np.nan
    
    if x == 0:
        if y > 0:
            ang = 0
        elif y < 0:
            ang = 180
        else:
            print('Both vectors are zero length, no angle calculated')
            ang = np.nan
    elif x < 0: 
        rc = y / x
        ang = 90 - (np.arctan(rc) * (180 / np.pi))
        ang += 180
    elif x > 0: 
        rc = y / x
        ang = 90 - (np.arctan(rc) * (180 / np.pi))

    return ang


def angle_difference(a, b):
    c = a - b
    if c > 180:
        c -= 360
    return c


u_s = angledata['u_s1'][:26]
v_s = angledata['v_s1'][:26]
angles = []
angle_diffs = []

ship_headings = angledata['Turb_heading']

for i, _ in enumerate(u_s):
    u = u_s[i]
    v = v_s[i]
    angle = rotation(u, v)
    angles.append(angle)
    ship_heading = ship_headings[i]
    angle_diff = angle_difference(angle, ship_heading)
    angle_diffs.append(angle_diff)
    print(angle_diff)
    
#%% Plotting
    
# plt.figure()

# plt.errorbar(angle_diffs, angledata['Turb_kelvin_angle'], angledata['Turb_kelvin_angle_std'] / np.sqrt(5), linestyle='None', fmt='o', capsize=5)
# plt.hlines(19.47,-190,190, linestyles='dashed', colors='orange')
# plt.grid(True)
# plt.xlabel('Angle between current and travel direction (deg)')
# plt.ylabel('Turbulent Kelvin Angle (deg)')
# plt.tight_layout()
# plt.xlim(-183,183)
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))

# plt.show()

# #%%  Plot asymmetry

# plt.figure()

# plt.errorbar(angledata['ship_current_angle'], angledata['angle_dif'], angledata['angle_dif_std'] , linestyle='None', fmt='o', capsize=5)
# plt.grid(True)
# plt.xlabel('Angle between current and travel direction (deg)')
# plt.ylabel('Asymmetry between Kelvin arms (deg)')
# plt.tight_layout()
# plt.xlim(-183,183)
# # plt.legend(['Theoretical Kelvin angle', 'Measurements'])

# plt.show()
    
#%% Plot water depth
    
plt.figure()

plt.errorbar(-angledata['Depth'], angledata['Turb_kelvin_angle'], angledata['Turb_kelvin_angle_std'] , linestyle='None', fmt='o', capsize=5)
plt.hlines(19.3,-190,190, linestyles='dashed', colors='orange')
plt.grid(True)
plt.xlabel('Water depth (m)')
plt.ylabel(r'$\beta $(deg)')
plt.tight_layout()
plt.xlim(0,100)
plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))

plt.show()