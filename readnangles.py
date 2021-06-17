# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

filepath = r'C:\Users\Ruben\Documents\Thesis\Data\Angles2\Results\res'
datapath = r'C:\Users\Ruben\Documents\Thesis\Data\Processing\shipdata.csv'
newpath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_complete.csv'


files = os.listdir(filepath)
shipdata = pd.read_csv(datapath)
newdata = pd.read_csv(newpath)

N = 26

standard_angle = [19.5] * 26

n = [i + 1 for i in range(N)]

names = list(np.zeros(N))


kelvin_angles = np.zeros((N, 12))
turb_kelvin_angles = np.zeros((N, 12))
direction = np.zeros((N, 12))
turb_direction = np.zeros((N, 12))


for i, file in enumerate(files):
    path = filepath + '/' + file 
    with open(path, 'rb') as f:
        data = pickle.load(f)

    
    for j, ship in enumerate(data):
        names[j] = ship[0]
        kelvin_angles[j, i] = ship[1]
        turb_kelvin_angles[j, i] = ship[2]
        direction[j, i] = ship[3]
        turb_direction[j, i] = ship[4]


#Rename names to correct format
for i, name in enumerate(names):
    if name[0] == '9':
        name = name[0:4]
    else:
        name = name[0:5]    
    names[i] = name    


# print(kelvin_angles)

# kelvin_angles = np.sort(kelvin_angles)

# print(kelvin_angles)

kelvin_angles_avg = np.mean(kelvin_angles, axis=1)
kelvin_angles_std = np.std(kelvin_angles, axis=1)
turb_kelvin_angles_avg = np.mean(turb_kelvin_angles, axis=1)
turb_kelvin_angles_std = np.std(turb_kelvin_angles, axis=1)
headings = np.mean(direction, axis=1)
turb_headings = np.mean(turb_direction, axis=1)
  

print(f'The mean Kelvin angle is: {np.mean(kelvin_angles_avg)}')
print(f'The mean turbulent Kelvin angle is: {np.mean(turb_kelvin_angles_avg)}')
print(f'The mean Kelvin angle std is: {np.mean(kelvin_angles_std)}')
print(f'The mean turbulent Kelvin angle std is: {np.mean(turb_kelvin_angles_std)}')


#%% Plotting

# plt.figure()

# plt.errorbar(n, kelvin_angles_avg, kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
# plt.plot(n, standard_angle)
# plt.xlabel('Ship number', fontsize=14)
# plt.ylabel('Kelvin Angle (degrees)', fontsize=14)
# plt.grid(True)
# # plt.title('Kelvin Angle')

# plt.show()

# plt.figure()

# plt.errorbar([i + 1 for i in range(26)], turb_kelvin_angles_avg, turb_kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
# plt.plot(n, standard_angle)
# plt.xlabel('Ship number', fontsize=14)
# plt.ylabel('Turbulent Kelvin Angle (degrees)', fontsize=14)
# plt.title('Turbulent Kelvin Angle')
           
# plt.grid(True)


# plt.figure()

# plt.errorbar(kelvin_angles_avg, turb_kelvin_angles_avg, kelvin_angles_std, turb_kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
# plt.plot([12,22], [12,22])
# plt.xlabel('Kelvin Angle (degrees)', fontsize=14)
# plt.ylabel('Turbulent Kelvin Angle (degrees)', fontsize=14)
# # plt.title('')
           
# plt.grid(True)

# plt.show()


# plt.scatter(newdata['length'], kelvin_angles_std)
# plt.xlabel('Ship Length (m)', fontsize=14)
# plt.ylabel('Kelvin Angle standard deviation', fontsize=14)
# # plt.title('')
           
# plt.grid(True)

# plt.show()

# #Expand csv with angles
# shipdata['Kelvin_angle'] = kelvin_angles_avg
# shipdata['Kelvin_angle_std'] = kelvin_angles_std
# shipdata['Turb_kelvin_angle'] = turb_kelvin_angles_avg
# shipdata['Turb_kelvin_angle_std'] = turb_kelvin_angles_std
# shipdata['Heading'] = headings
# shipdata['Turb_heading'] = turb_headings

# shipdata.to_csv(r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_angles.csv')


