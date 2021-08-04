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

u_o = angledata['obs_cur1'][:26]
v_o = angledata['obs_cur2'][:26]



u_s = angledata['u_s1'][:26]
v_s = angledata['v_s1'][:26]
angles = []
angle_diffs = []
obs_angles = []
obs_angle_diffs = []
angle_perp = []
obs_angle_perp = []

ship_headings = angledata['Turb_heading']

for i, _ in enumerate(u_s):
    u = u_s[i]
    v = v_s[i]
    x = u_o[i]
    y = v_o[i]
    s = np.sqrt(u**2 + v**2)
    s_o = np.sqrt(x**2 + y**2)
    
    
    
    obs_angle = rotation(x,y)
    angle = rotation(u, v)
    angles.append(angle)
    obs_angles.append(obs_angle)
    ship_heading = ship_headings[i]
    angle_diff = angle_difference(angle, ship_heading)
    obs_angle_diff = angle_difference(obs_angle, ship_heading)
    angle_diffs.append(angle_diff)
    obs_angle_diffs.append(obs_angle_diff)
    
    s_p = s * np.sin(angle_diff * np.pi/180)
    s_o_p = s_o * np.sin(obs_angle_diff * np.pi/180)
    angle_perp.append(s_p)
    obs_angle_perp.append(s_o_p)
    # print(s_p, s, angle_diff)
    # print(angle_diff)


   
#%% Plotting


    
# plt.figure()

# plt.errorbar(angle_diffs, angledata['Turb_kelvin_angle'], angledata['Turb_kelvin_angle_std'] / np.sqrt(5), linestyle='None', fmt='o', capsize=5)
# plt.hlines(19.47,-190,190, linestyles='dashed', colors='green')
# plt.grid(True)
# plt.xlabel('Angle between current and travel direction (deg)')
# plt.ylabel('Turbulent Kelvin Angle (deg)')
# plt.tight_layout()
# plt.xlim(-183,183)
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))

# plt.show()

# # #%%  Plot asymmetry



# plt.figure()

# plt.errorbar(angledata['ship_current_angle'], angledata['angle_dif'], angledata['angle_dif_std'] , linestyle='None', fmt='o', capsize=5)
# plt.grid(True)
# plt.xlabel(r'$\theta_{current}$ (deg)')
# plt.ylabel(r'$\alpha$ (deg)')
# plt.tight_layout()
# plt.xlim(-183,183)
# # plt.legend(['Theoretical Kelvin angle', 'Measurements'])

# plt.show()

#%%

matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 20

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharey=True)

# plt.figure(figsize=(2000, 6), dpi=80)

ax1.errorbar(angledata['perp_current'], angledata['angle_dif'], angledata['angle_dif_std'], linestyle='None', fmt = 'o', capsize = 5)
# ax1.set_xlim(-1,1)

ax1.grid()
ax1.set_xticks([-0.2,-0.1,0,0.1])
ax1.set_xlabel(r'$c_{\perp}$ (m/s)', fontsize=24)
ax1.set_ylabel(r'$\bar{\alpha} (deg)$', fontsize=24)
ax1.text(0.05,5,'(a)', fontsize=24)

ax2.errorbar(angledata['obs_perp_current'], angledata['angle_dif'], angledata['angle_dif_std'], linestyle='None', fmt = 'o', capsize = 5)

# ax2.set_xlim(-180,180)
ax2.set_xticks([-0.125,-0.1,-0.075,-0.05])
# ax2.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)), fontsize=24)
ax2.grid()
ax2.set_xlabel(r'$\hat{c}_{\perp}$ (m/s)', fontsize=24)
ax2.text(-0.063,5,'(b)', fontsize=24)

plt.show()
    
#%% Plot water depth
    
# plt.figure()

# plt.errorbar(-angledata['Depth'], angledata['Turb_kelvin_angle'], angledata['Turb_kelvin_angle_std'] , linestyle='None', fmt='o', capsize=5)
# plt.hlines(19.3,-190,190, linestyles='dashed', colors='orange')
# plt.grid(True)
# plt.xlabel('Water depth (m)')
# plt.ylabel(r'$\beta $(deg)')
# plt.tight_layout()
# plt.xlim(0,100)
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))

# plt.show()