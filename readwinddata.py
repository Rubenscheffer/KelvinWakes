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
        'size': 16}

matplotlib.rc('font', **font)

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

#%% Main

datapath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/winddata.csv'
newpath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_complete.csv'

angledata = pd.read_csv(newpath)

data = pd.read_csv(datapath)


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


u_w = data['u_w'][:26]
v_w = data['v_w'][:26]
angles = []
angle_diffs = []
perps = []

ship_headings = angledata['Turb_heading']

for i, _ in enumerate(u_w):
    u = u_w[i]
    v = v_w[i]
    s = np.sqrt(u**2 + v**2)
    angle = rotation(u, v)
    angles.append(angle)
    ship_heading = ship_headings[i]
    angle_diff = angle_difference(angle, ship_heading)
    angle_diffs.append(angle_diff)
    p = s * np.sin(angle_diff*np.pi/180)
    perps.append(p)
    print(angle_diff)
    
#%% Plotting
    
plt.figure()

plt.errorbar(angle_diffs, angledata['Turb_kelvin_angle'], angledata['Turb_kelvin_angle_std'], linestyle='None', fmt='o', capsize=5)
plt.hlines(19.47,-190,190, linestyles='dashed', color='green')
plt.grid(True)
plt.xlabel(r'$\theta_w$ (deg)')
plt.ylabel(r'$\beta (deg)$')
plt.tight_layout()
plt.xlim(-183,183)
plt.legend(['Theoretical Kelvin angle', 'Measurements'], loc=((1.04,0.5)))

plt.show()

#%% Select angles

#%% Plot assymetry

plt.figure()

plt.errorbar(angledata['perp_wind'], angledata['angle_dif'], angledata['angle_dif_std'] , linestyle='None', fmt='o', capsize=5)
plt.grid(True)
plt.xlabel(r'$w_{\perp}$ (m/s)')
plt.ylabel(r'$\bar{\alpha}$ (deg)')
plt.tight_layout()
plt.xlim(-5,-1)
# plt.legend(['Theoretical Kelvin angle', 'Measurements'])

plt.show()