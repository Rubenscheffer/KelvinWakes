# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:54:14 2021

@author: Ruben
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

with open(r'C:/Users/Ruben/Documents/Thesis/Schrijven/data.pickle', 'rb') as file:
    data = pickle.load(file) 
#%%  Setup


z = StandardScaler()
EM = GaussianMixture(n_components=3)


#%% Calculation

N = 26
#Empte arrays
sil_scores = np.zeros(N)


for i in range(N):

    test = np.array(data[i,:])
    test = np.reshape(test, (-1,1))
    test = z.fit_transform(test)
    EM.fit(test)
    
    cluster = EM.predict(test)
    cluster_p = EM.predict_proba(test)
    sil_score = silhouette_score(test, cluster)
    sil_scores[i] = sil_score

    
    # if i == 19:
    #     print(cluster_p)
    #     print(f'Silhoutte score:{sil_score}')
    
#%% Plot
    
plt.figure()

for i in range(12):
    plt.scatter([j for j in range(26)], data[:,i], color='blue',s=14)
    
    
plt.grid()    

plt.show()