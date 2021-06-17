# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:55:52 2021

@author: Ruben
"""


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

newpath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_complete.csv'

newdata = pd.read_csv(newpath)

plt.figure()

plt.scatter(newdata.length, newdata.Turb_kelvin_angle_std)

plt.xlabel('speed (knots)', fontsize=14)
plt.ylabel('Turbulent Kelvin angle (degrees)', fontsize=14)


plt.grid(True)

plt.show()




# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2)

# ax1[0].errorbar(newdata.speed, newdata.Turb_kelvin_angle, newdata.Turb_kelvin_angle_std, linestyle='None', fmt = 'o', capsize = 5)

# ax1[0].set_xlabel('Speed (knots)', fontsize=14)
# ax1[0].set_ylabel('Turbulent Kelvin angle (degrees)', fontsize=14)


# ax1[0].grid(True)

# plt.show()





plt.figure()

plt.errorbar(newdata.speed, newdata.Turb_kelvin_angle, newdata.Turb_kelvin_angle_std, linestyle='None', fmt = 'o', capsize = 5)

plt.xlabel('Speed (m/s)', fontsize=14)
plt.ylabel('Turbulent Kelvin angle (degrees)', fontsize=14)


plt.grid(True)

plt.show()




plt.figure()

plt.errorbar(newdata.length, newdata.Turb_kelvin_angle, newdata.Turb_kelvin_angle_std, linestyle='None', fmt = 'o', capsize = 5)

plt.xlabel('Ship length (meters)', fontsize=14)
plt.ylabel('Turbulent Kelvin angle (degrees)', fontsize=14)


plt.grid(True)

plt.show()


plt.figure()

plt.errorbar(newdata.width, newdata.Turb_kelvin_angle,  newdata.Turb_kelvin_angle_std, linestyle='None', fmt = 'o', capsize = 5)

plt.xlabel('Ship width (m)', fontsize=14)
plt.ylabel('Turbulent Kelvin angle (degrees)', fontsize=14)


plt.grid(True)

plt.show()


plt.figure()

plt.errorbar(newdata.draught, newdata.Turb_kelvin_angle,  newdata.Turb_kelvin_angle_std, linestyle='None', fmt = 'o', capsize = 5)

plt.xlabel('Draught (m)', fontsize=14)
plt.ylabel('Turbulent Kelvin angle (degrees)', fontsize=14)


plt.grid(True)

plt.show()



plt.figure()

plt.errorbar(-newdata.Depth1, newdata.Turb_kelvin_angle,  newdata.Turb_kelvin_angle_std, linestyle='None', fmt = 'o', capsize = 5)

plt.xlabel('Water Depth (m)', fontsize=14)
plt.ylabel('Turbulent Kelvin angle (degrees)', fontsize=14)


plt.grid(True)

plt.show()