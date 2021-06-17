# -*- coding: utf-8 -*-
"""
Program to add theoretical Froude number to data

@author: Ruben
"""
# import pickle as pi
import pandas as pd
import numpy as np
# import os


datapath = r'C:\Users\Ruben\Documents\Thesis\Data\Processing\shipdata_complete.csv'

shipdata = pd.read_csv(datapath)

shipdata_ais = shipdata.loc[shipdata.draught > 0]

froude = np.zeros(11)
g = 9.81

i = -1

for _, ship in shipdata_ais.iterrows():
    i += 1
    v = ship.speed * 0.51444  #knots to m/s
    length = ship.length
    froude[i] = v / (np.sqrt(length * g))
    print(i)
    print(froude)
    
shipdata_ais['froude'] = froude

shipdata_ais.to_csv(r'C://Users/Ruben/Documents/Thesis/Data/Processing/shipdata_aisfroude.csv')