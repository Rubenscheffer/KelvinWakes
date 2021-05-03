# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:45:23 2021

@author: Ruben
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(r'C:\Users\Ruben\Documents\Thesis\Data\Angles2\Results\reduced2.p', 'rb') as f:
    data = pickle.load(f)
 
angles,turb = [], []


#Data analysis

for i, dat in enumerate(data):
    
    angles.append(dat[1])
    turb.append(dat[2])
    # print(dat[1])
    # print(dat[2])
print(np.mean(angles))
print(np.mean(turb))

plt.figure()
plt.hist(angles,bins=50)
plt.show()