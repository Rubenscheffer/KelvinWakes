# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import pickle

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
n_2 = [i + 1.2 for i in range(N)]

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

#%% Save turb_kelvin angles

with open('C:/Users/Ruben/Documents/Thesis/Schrijven/data.pickle', 'wb') as file:
    pickle.dump(turb_kelvin_angles[:], file)
   
    





 
#%%
    
turb_copy = pd.DataFrame(turb_kelvin_angles)
# print(kelvin_angles)

# kelvin_angles = np.sort(kelvin_angles)

# print(kelvin_angles)

# print(kelvin_angles[:,:5])

kelvin_angles_avg_1 = np.mean(kelvin_angles[:,:5], axis=1)
kelvin_angles_std_1 = np.std(kelvin_angles[:,:5], axis=1)
turb_kelvin_angles_avg_1 = np.mean(turb_kelvin_angles[:,:5], axis=1)
turb_kelvin_angles_std_1 = np.std(turb_kelvin_angles[:,:5], axis=1)
headings_1 = np.mean(direction[:,:5], axis=1)
turb_headings_1 = np.mean(turb_direction[:,:5], axis=1)

indexes = np.argsort(turb_kelvin_angles_avg_1)

kelvin_angles_avg_2 = np.mean(kelvin_angles[:,5:10], axis=1)
kelvin_angles_std_2 = np.std(kelvin_angles[:,5:10], axis=1)
turb_kelvin_angles_avg_2 = np.mean(turb_kelvin_angles[:,5:10], axis=1)
turb_kelvin_angles_std_2 = np.std(turb_kelvin_angles[:,5:10], axis=1)
headings_2 = np.mean(direction[:,5:10], axis=1)
turb_headings_2 = np.mean(turb_direction[:,5:10], axis=1)

kelvin_angles_avg = np.mean(kelvin_angles, axis=1)
kelvin_angles_std = np.std(kelvin_angles, axis=1)
turb_kelvin_angles_avg = np.mean(turb_kelvin_angles, axis=1)
turb_kelvin_angles_std = np.std(turb_kelvin_angles, axis=1)

#Sort for individual points
all_indexes = np.argsort(turb_kelvin_angles_avg)

print(np.mean(turb_kelvin_angles_std_1))
print(np.mean(turb_kelvin_angles_std_2))

turb_copy['index'] = all_indexes

turb_sorted = turb_copy.sort_values('index',axis=0)

headings = np.mean(direction, axis=1)
turb_headings = np.mean(turb_direction, axis=1)
  
tot_data = list(zip(turb_kelvin_angles_avg_1, turb_kelvin_angles_std_1, turb_kelvin_angles_avg_2, turb_kelvin_angles_std_2, turb_kelvin_angles[:,10], turb_kelvin_angles[:,11]))

tot_frame = pd.DataFrame(tot_data, columns=['avg1','std1','avg2','std2','p3', 'p4'])


sorted_tot = tot_frame.sort_values('avg1')


#%% Analysis

turb_kelvin_angles_avg_1 = np.array(turb_kelvin_angles_avg_1)
turb_kelvin_angles_avg_2 = np.array(turb_kelvin_angles_avg_2)

delta_th_1 = np.mean(abs(turb_kelvin_angles_avg_1-19.47))
delta_th_2 = np.mean(abs(turb_kelvin_angles_avg_2-19.47))

delta = np.mean(abs(turb_kelvin_angles_avg_1 - turb_kelvin_angles_avg_2))

print(delta_th_1)
print(delta)
print(delta_th_2)
# print(f'The mean Kelvin angle is: {np.mean(kelvin_angles_avg)}')
# print(f'The mean turbulent Kelvin angle is: {np.mean(turb_kelvin_angles_avg)}')
# print(f'The mean Kelvin angle std is: {np.mean(kelvin_angles_std)}')
# print(f'The mean turbulent Kelvin angle std is: {np.mean(turb_kelvin_angles_std)}')
#%% Sort kelvin angles
    
total_turb = np.zeros((26,3))
total_turb[:,0] = turb_kelvin_angles_avg
total_turb[:,1] = turb_kelvin_angles_std
total_turb[:,2] = indexes

print(np.shape(total_turb))

dataframe = pd.DataFrame(total_turb, columns=["avg", "std","index"])
sorted_frame = dataframe.sort_values('index')


#%% Plotting


# # social angles
# plt.figure()

# for i in range(12):
#     plt.scatter(n, turb_kelvin_angles[:,i], color='blue',s=14)

# # plt.errorbar(n, sorted_tot['avg1'], sorted_tot['std1'], linestyle='None', fmt = 'o', capsize = 3, color='blue')
# # plt.errorbar(n_2, sorted_tot['avg2'], sorted_tot['std2'], linestyle='None', fmt = 'o', capsize = 3, color='red')
# # plt.scatter(n, sorted_tot['p3'])
# # plt.scatter(n, sorted_tot['p4'])



# plt.plot([-10,50], [19.47,19.47], '--', color='green')
# plt.xlabel('Ship number', fontsize=14)
# plt.ylabel(r'$\beta (deg)$', fontsize=14)
# x_ticks = [5,7,11,15,20,25]
# plt.xticks(ticks=x_ticks)


# plt.xlim(0.5,26.5)
# # plt.legend(['Theoretical value', 'Person 1','Person 2','Own measurement','Person 3'], loc=((1.04,0.5)))
# plt.grid(True)

# plt.show()

#%%

#first angles
# plt.figure()

# plt.errorbar(n, sorted_tot['avg1'], sorted_tot['std1'], linestyle='None', fmt = 'o', capsize = 3, color='blue')
# # plt.errorbar(n_2, sorted_tot['avg2'], sorted_tot['std2'], linestyle='None', fmt = 'o', capsize = 3, color='red')
# # plt.scatter(n, sorted_tot['p3'])
# # plt.scatter(n, sorted_tot['p4'])
# plt.plot(n, standard_angle, '--', color='green')
# plt.xlabel('Ship number', fontsize=14)
# plt.ylabel(r'$\bar{\beta} (deg)$', fontsize=14)
# plt.xlim(0.5,26.5)
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))
# plt.grid(True)

# plt.show()

# print(np.mean(sorted_tot['std1']))

sorted_frame = sorted_frame.sort_values('avg')

#average angles
#%%
plt.figure()

plt.errorbar([i + 1 for i in range(26)], sorted_frame['avg'], sorted_frame['std'], linestyle='None', fmt = 'o', capsize = 5)
plt.plot(n, standard_angle, '--', color='green')
plt.xlabel('Ship number', fontsize=14)
plt.ylabel(r'$\bar{\beta} (deg)$', fontsize=14)
# plt.title('Turbulent Kelvin Angle')
plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))
plt.xlim(0.5,26.5)
plt.grid(True)

x_ticks = [2,6,7,11,16,20,25]
plt.xticks(ticks=x_ticks)


plt.show()

#%%Random plots


#Currents and wind correlations

#Line

a, b = np.polyfit(newdata['current_heading'][:-1], newdata['wind_angle'][:-1], 1)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))

ax1.scatter(newdata['current_heading'], newdata['wind_angle'])
ax2.scatter(newdata['obs_current_heading'], newdata['wind_angle'])
# ax1.plot([-400,400], [19.47,19.47], '--', color='green')
ax1.set_xlim(0,360)
ax1.set_xticks([0,90,180,270,360])
ax1.set_yticks([0,60,120,180,240,300,360])
ax1.set_ylim(0,360)
ax2.set_xlim(0,360)
ax2.set_xticks([0,90,180,270,360])
ax2.set_yticks([0,60,120,180,240,300,360])
ax2.set_ylim(0,360)

# ax2.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))
ax1.plot([0,360],[b,a*360+b], linestyle='--', color='g')
ax1.text(90,110,'y = 0.67x + 70')
ax1.text(300,20,'(a)')
ax2.text(300,20,'(b)')
ax1.grid()
ax1.set_xlabel(r'$\gamma_{current}$ (deg)', fontsize=16)
ax1.set_ylabel(r'$\gamma_{wind}$ (deg)', fontsize=16)
ax2.grid()
ax2.set_xlabel(r'$\hat{\gamma}_{current}$ (deg)', fontsize=16)

# plt.figure()

# plt.errorbar(newdata['ship_wind_angle'], turb_kelvin_angles_avg, turb_kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
# plt.plot([-400,400], [19.47,19.47], '--', color='green')
# plt.xlabel(r'$\theta_{wind}$ (deg)', fontsize=14)
# plt.ylabel(r'$\bar{\beta} (deg)$', fontsize=14)
# plt.xlim(-180,180)
# plt.xticks([-180,-90,0,90,180])
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)), fontsize=14)
# # plt.title('')
           
# plt.grid(True)

# plt.show()


# # categories = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,0, 0,0, 0,0, 0,1, 0,0, 0,0, 0,0, 0,0, 0,])

# # colormap = np.array(['blue', 'red', 'green'])

#%% Double plot
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 20
    
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharey=True)

# plt.figure(figsize=(2000, 6), dpi=80)

ax1.errorbar(newdata['ship_current_angle'], turb_kelvin_angles_avg, turb_kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
ax1.plot([-400,400], [19.47,19.47], '--', color='green')
ax1.set_xlim(-180,180)
# ax2.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))
ax1.grid()
ax1.set_xticks([-180,-90,0,90,180])
ax1.set_xlabel(r'$\theta_{current}$ (deg)', fontsize=24)
ax1.set_ylabel(r'$\bar{\beta} (deg)$', fontsize=24)


ax2.errorbar(newdata['obs_current_dif'], turb_kelvin_angles_avg, turb_kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
ax2.plot([-400,400], [19.47,19.47], '--', color='green')
ax2.set_xlim(-180,180)
ax2.set_xticks([-180,-90,0,90,180])
# ax2.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)), fontsize=24)
ax2.grid()
ax2.set_xlabel(r'$\hat{\theta}_{current}$ (deg)', fontsize=24)


plt.show()



# # plt.figure()

# plt.errorbar(newdata['froude'], turb_kelvin_angles_avg, turb_kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
# plt.xlabel(r'$Fr_L$ (m)', fontsize=14)

# # plt.ylabel(r'$\bar{\beta} (deg)$', fontsize=14)
# plt.plot([-400,400], [19.47,19.47], '--', color='green')
# plt.xlim(0,0.2)
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))

# ax = plt.gca()
# # ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_ticklabels([])
     
# plt.grid(True)

# plt.show()

# delta = -newdata['length']/newdata['Depth']
# delta = delta.dropna()
# delta = delta.sort_values()

# frh = newdata['Froude depth']
# frh = frh.dropna()
# frh = frh.sort_values()

# plt.figure()

# plt.scatter([i + 1 for i in range(11)], frh)
# plt.xlabel(r'Ship number', fontsize=14)
# plt.ylabel(r'$Fr_H$', fontsize=14)
# plt.xlim(0.5,11.5)
# plt.hlines(1, 0, 15, linestyles='--', color = 'g')
# plt.legend(['Measurements', r'$Fr_{H,crit}$'], loc=((1.04,0.5)))
# # plt.title('')
           
# plt.grid(True)

# plt.show()


#%% reserve


# plt.figure()

# plt.errorbar(-newdata['Depth'], turb_kelvin_angles_avg, turb_kelvin_angles_std, linestyle='None', fmt = 'o', capsize = 5)
# plt.errorbar(-newdata['Depth'][9], turb_kelvin_angles_avg[9], turb_kelvin_angles_std[9], linestyle='None', fmt = 'o', capsize = 5, c='red')
# plt.errorbar(-newdata['Depth'][16], turb_kelvin_angles_avg[16], turb_kelvin_angles_std[16], linestyle='None', fmt = 'o', capsize = 5, c='red')
# plt.xlabel(r'$\theta_{wind}$ (deg)', fontsize=14)
# plt.ylabel(r'$\beta (deg)$', fontsize=14)
# plt.plot([0,400], [19.47,19.47], '--', color='green')
# plt.xlim(0,360)
# plt.legend(['Theoretical value', 'Measurements'], loc=((1.04,0.5)))
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


