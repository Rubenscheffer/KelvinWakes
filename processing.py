# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:34:38 2021

@author: Ruben
"""

import pickle as pi
import pandas as pd

savepath = r'C://Users/Ruben/Documents/Thesis/Data/Processing/AISdataframe.p'
datapath = r'C:\Users\Ruben\Documents\Thesis\Data\Processing\shipdata.csv'

ais = pi.load(open(savepath, 'rb'))

shipdata = pd.read_csv(datapath)

# print(ais['latitude'].min())

#loop through ships, couple each ship to AIS ship.

for i, ship in shipdata.iterrows():
    location = ship.Location
    date = ship.Date
    time = ship.Time
    

#Transfor data to match AIS data format
    lat, lon = location.split(',')
    lat, lon = float(lat), float(lon)
    date = date.split('/')[2]
    hour, minute, second = time.split(':')
    print(lat,lon)
    
#Filter AIS data to find list of possible ships

    data = ais.copy(deep=True)
    data = data.loc[(data.date == float(date))]
    #Assume datapoint should exist within 15 minutes of image
    #Assume max speed is 25kts
    max_v = 25 * 1.85
    # print(f'Before: {data.shape}')
    #Convert from fraction to arcminutes/seconds properly
    data = data.loc[(data.latitude < lat + 0.1)]
    # print(f'After: {data.shape}')
    # data = data.loc[(data.latitude < float(lat) + 0.1) & (data.latitude > float(lat) - 0.1)]
    
#Use function that minimizes distance/time
