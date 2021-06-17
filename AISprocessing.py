# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:34:38 2021

@author: Ruben
"""

import pickle as pi
import pandas as pd
import numpy as np
import os

#TODO
#Fix hour jump
#Pick from best options
#--> Current option, compare closest up to 2x the closest
#--> Other option, compare all within range r (1km?)
#Replace single point linear interpolation with multiple point bi-linear interp.

N = 26

filepath = r'C:\Users\Ruben\Documents\Thesis\Data\Angles2\Results\res'

files = os.listdir(filepath)

savepath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/AISdataframe.p'
datapath = r'C:\Users\Ruben\Documents\Thesis\Data\Processing\shipdata_angles.csv'

ais = pi.load(open(savepath, 'rb'))

shipdata = pd.read_csv(datapath)


#loop through ships, couple each ship to AIS ship.


#Create some empty lists
lowest_distances = np.zeros((len(shipdata), 2))
filtered_list = []
ship_headings = np.zeros(N)


for i, ship in shipdata.iterrows():
    location = ship.Location
    date = ship.Date
    time = ship.Time
    ship_headings[i] = ship.Heading  #Heading based on ship
    print('\n')
    print(f'Ship {i + 1}', sep='\n')
   
#Transform data to match AIS data format
    lon, lat = location.split(',')
    lat, lon = float(lat), float(lon)
    date = date.split('/')[2]
    hour, minute, second = time.split(':')
    print(f'{hour}:{minute}:{second}')
    
    # print(lat,lon)
    
    # print(lat)
    
    
#Filter AIS data to find list of possible ships
    
    
    data = ais.copy(deep=True)
    data = data.loc[(data.date == float(date))]

   
    print(f'shape before filters: {data.shape}')
    #Location filter
    data = data.loc[(data.latitude < (lat + 0.15)) & (data.latitude > lat - 0.15)]
    data = data.loc[(data.longitude < (lon + 0.15)) & (data.longitude > lon - 0.15)]
    print(f'shape after location filter: {data.shape}')
    
    #Time filter

    # data = data.loc[(data.hours == int(hour))]
    # data = data.loc[(data.minutes - int(minute) < 20)]
    
    print(f'shape after time filter: {data.shape}')
    
    #Further filter on basis of heading, course and velocity
    data = data.loc[(data.course < 361)]
    data = data.loc[(data.heading < 361)]
    data = data.loc[(data.speed > 0.2)]
    
    print(f'shape after heading/velocity filter: {data.shape}')
    

    #Use function that minimizes distance/time
    interpolated_dis = []
    
    

    
    for j, dat in data.iterrows():
        # delta_x = abs(dat.longitude - lon) * 111
        # delta_y = abs(dat.latitude - lat) * 111
        # dis = np.sqrt(delta_x**2 + delta_y**2)
        
        
        # projection with course and velocity
        t_i = dat.minutes  #In minutes from nearest hour
        delta_t = t_i - int(minute)
        x_i = dat.longitude
        y_i = dat.latitude
        v_i = dat.speed * 1.85 / (60 * 111)  #in deg/min
        h_i = dat.course  #in degrees
        
        #Create interpolation point
        
        
        x_int = x_i + delta_t * v_i * np.sin(h_i) 
        y_int = y_i + delta_t * v_i * np.cos(h_i)
        
        
        # print(f'heading: {h_i}, velocity: {dat.speed} and delta_t: {delta_t}')
        # print(f'Interpolated coordinates: ({x_int},{y_int})')
        
        #Measure distance between actual location and interpolation and save
        
        dis = np.sqrt(((x_int - lon) * 111) ** 2 + ((y_int - lat) * 111) ** 2)  #distance in km
        interpolated_dis.append([dis, dat.mmsi, dat.heading, dat.speed])
            
    
    # print(interpolated_dis)
    interpolated_dis = sorted(interpolated_dis)  # sort
    # print(interpolated_dis)
    
    
    counted = []
    
    for r, item in enumerate(interpolated_dis):
        # Condition for filtered list, now only contains data under 1km dist.
        if item[0] > 1:
            break
        else: 
            counted.append([item[0], item[1], item[2], item[3]])
   
    filtered_list.append(counted)
    # filtered_list = np.unique(filtered_list)
    
    #Ships found: [2,4,5,6,9,10,11,17,23,27]
    #No data: [24, 25]
    #Distance too high: [0, 1, 3, 12, 22]
    #Multiple options: [7, 13, 14, 15, 16, 18, 19 ,20, 21, 26, 27, 28]
    
    
    # if len(interpolated_dis) != 0:
    #     lowest_distances[i] = interpolated_dis[0][0], interpolated_dis[0][1]

#%%Check filtered list for headings
      
        
def dot_product(x1, x2, y1, y2):
    return x1 * x2 + y1 * y2
    


def deg_to_rad(x):
    return x * (np.pi / 180)
  


ais_couple = np.zeros((N,5))
ais_couple[:,:] = np.nan
     
for i, item in enumerate(filtered_list):
    
    ais_couple[i,0] = i + 1
    
    if len(item) == 0:
        continue
    else:
        
        #Load heading from picture
        print('\n')
        print(f'Ship {i+1}')
        print(f'Heading is: {ship_headings[i]}')
        x_heading = np.sin(deg_to_rad(ship_headings[i]))
        y_heading = np.cos(deg_to_rad(ship_headings[i]))
        
        
        #Loop through possible AIS points of ship
        for j, point in enumerate(item):
            # print(f'AIS heading is: {point[2]} with distance: {point[0]}')
            unit_vector = 1
            # print(point)
            x_vector = np.sin(deg_to_rad(point[2]))
            y_vector = np.cos(deg_to_rad(point[2]))
            
            vector_angle = (180 / np.pi) * np.arccos(dot_product(x_heading, x_vector, y_heading, y_vector))
            # print(f'Angle between vectors: {vector_angle}')
            
            if vector_angle < 30:
                print(f'AIS heading is: {point[2]} with distance: {point[0]}')
                ais_couple[i,1] = point[1]
                ais_couple[i,2] = point[0]
                ais_couple[i,3] = vector_angle
                ais_couple[i,4] = point[3]
                
                
                
                
                
                
                break


#%% Couple AIS data to dataset    

#Empty arrays to add to shipdata
ship_length = np.full(26, np.nan)
ship_speed = np.full(26, np.nan)
ship_draught = np.full(26, np.nan)
ship_width = np.full(26, np.nan)


#Start loop to fill arrays
for i, ship in enumerate(ais_couple):        
    data = ais.copy(deep=True)
    data = data.loc[data.mmsi == ship[1]]
    if len(data) > 0:
        data = data.iloc[0]
        ship_length[i] = data.length
        ship_speed[i] = ship[4]
        ship_draught[i] = data.draught
        ship_width[i] = data.width

shipdata['speed'] = ship_speed
shipdata['length'] = ship_length
shipdata['width'] = ship_width
shipdata['draught'] = ship_draught

#%% Angle between current and ship





# shipdata.to_csv(r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_complete.csv')