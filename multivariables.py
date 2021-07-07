# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:18:34 2021

@author: Ruben
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C://Users/Ruben/Documents/Thesis/Data/Processing/multivariables.csv', index_col=0)
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()