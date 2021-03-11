# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:35:13 2021

@author: Ruben
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import zipfile

folder = r'C://Users/Ruben/Documents/Thesis/Data/Angles1/'

files = os.listdir(folder)


def json_reader(file):
    """
    Reads position and time from Planet .json metadata file
    
    input:  file      string      Path to .json file
    
    output: geometry  list        n x 2 list of coordinates describing imagery area
            time      string      UTC time of image
    """
    with open(path) as f:
        data = json.load(f)

    geometry = data['geometry']['coordinates'][0]
    time = data['properties']['acquired']

    return geometry, time



for i, file in enumerate(files):
    if file[-3:] == 'zip':
        path = folder + file
        zipfolder = zipfile.ZipFile(path, 'r')
        zipfiles = zipfolder.namelist()
        
        for files in zipfiles:
            if files[-9:] == 'data.json':
                geom, time = json_reader(files)
                print('check')
    
