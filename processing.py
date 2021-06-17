# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:34:38 2021

@author: Ruben
"""

import pickle as pi
import pandas as pd
import numpy as np

#Function to improve


def deg_to_dms(angle):
    angle = str(angle)
    # print(f'input:{angle}')
    angle = angle.split('.')
    dec = int(angle[1])
    num = int(angle[0])
    # print(f'num, dec:{num},{dec}')
    dec *= 0.6
    dec = int(dec)
    # print(f'{dec}')
    # print(f'{num}')
    # print(f'{dec}')
    new_angle = str(num) + '.' + str(dec)
    new_angle = float(new_angle)
    new_angle = round(new_angle, 5)
    # print(f'{new_angle}')
    return new_angle



savepath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/AISdataframe.p'
datapath = r'C:\Users\Ruben\Documents\Thesis\Data\Processing\shipdata.csv'

ais = pi.load(open(savepath, 'rb'))

shipdata = pd.read_csv(datapath)


#loop through ships, couple each ship to AIS ship.

for i, ship in shipdata.iterrows():
    location = ship.Location
    date = ship.Date
    time = ship.Time
    
    print(f'Ship {i + 1}')
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

   
    print(f'shape before location filter: {data.shape}')
    #Location filter
    data = data.loc[(data.latitude < (lat + 0.1)) & (data.latitude > lat - 0.1)]
    data = data.loc[(data.longitude < (lon + 0.1)) & (data.longitude > lon - 0.1)]
    print(f'shape after location filter: {data.shape}')
    
    #Time filter

    data = data.loc[(data.hours == int(hour))]
    data = data.loc[(data.minutes - int(minute) < 20)]
    
    print(f'shape after time filter: {data.shape}')
    
    #Further filter on basis of heading and velocity
    data = data.loc[(data.heading < 360)]
    data = data.loc[(data.speed > 0.2)]
    
    print(f'shape after heading/velocity filter: {data.shape}')
    

    #Use function that minimizes distance/time
    intepolated_dis = []
    ship_data = []
    
    
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
        h_i = dat.heading  #in degrees
        
        #Create interpolation point
        
        
        x_int = x_i + delta_t * v_i * np.sin(h_i) 
        y_int = y_i + delta_t * v_i * np.cos(h_i)
        
        # print(f'heading: {h_i}, velocity: {dat.speed} and delta_t: {delta_t}')
        # print(f'Interpolated coordinates: ({x_int},{y_int})')
        
        #Measure distance between actual location and interpolation and save
        
