# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:50:49 2021

@author: Ruben
"""

import json
import matplotlib.pyplot as plt
import numpy as np

folder = r'C://Users/Ruben/Documents/Thesis/Data/Angles1/Metadata/'
filename = 'ship1metadata.json'
path = folder + filename

with open(path) as f:
    data = json.load(f)

geometry = data['geometry']['coordinates'][0]
time = data['properties']['acquired']

print(geometry[0][0])

plt.figure()

x = []
y = []

for i, _ in enumerate(geometry):
    plt.scatter(geometry[i][0], geometry[i][1])
    x.append(geometry[i][0])
    y.append(geometry[i][1])

x = np.mean(x)
y = np.mean(y)
print(x,y)

plt.scatter(x,y)
plt.show()

