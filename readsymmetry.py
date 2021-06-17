# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:16:27 2021

@author: Ruben
"""


import pickle
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

filepath = r'C:\Users\Ruben\Documents\Thesis\Data\Angles2\Results\sym'

files = os.listdir(filepath)

for i, file in enumerate(files):
    filename = filepath + '/' + file

    with open(filename, 'rb') as f:
        data = pickle.load(f)