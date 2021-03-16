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
    with open(file) as f:
        data = json.load(f)

    geometry = data['geometry']['coordinates'][0]
    time = data['properties']['acquired']

    return geometry, time


def zip_search(file):
    """
    Searches folders and saves a list of all zip files within
    
    input: file      string      Path to folder
    
    output: zips     list        List of zip files
    """
    
    zips = []
    
    for i, doc in enumerate(file):
        if doc[-3:] == 'zip':
            path = folder + doc
            zips.append(path)
        
    return zips


def read_zip(zippath):
    """
    Read a single zip file and locate the metadata file within
    
    input: zippath      string      Path to zip file
    
    output: current_zip ZipFile     Zipfile object
    """
    current_zip = zipfile.ZipFile(zippath)
    files_in_zip = current_zip.namelist()\
    for i, file in enumerate(files_in_zip):
        if file[-13:] == 'metadata.json':
            print(file)
            print(type(current_zip.read(file)))
    return


zippaths = zip_search(files)
read_zip(zippaths[1])
