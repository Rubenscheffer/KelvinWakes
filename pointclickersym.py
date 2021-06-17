# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:06:43 2021

@author: Ruben
"""

import cv2
import numpy as np
import os
import pickle
# import sys

imagefolder = r'C:\Users\Ruben\Documents\Thesis\Data\Angles2\Symmetry'

files = os.listdir(imagefolder)

angles = []


def rotation(p1, p2):
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    rc = delta_y / delta_x
    ang = np.arctan(rc) * (180 / np.pi)
    direction = 90 + ang
    print(direction)
    if p1[0] < p2[0]:
        direction += 180
    return direction


for file in files:

    imagepath = imagefolder + '/' + str(file)
    
    
    #This variable we use to store the pixel location
    refPt = []
    
    
    #Read image and resize
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (0,0), fx=2, fy=2)
    
    #click event function
    
    def click_event(event, x, y, flags, param):
        if len(refPt) >= 7:
            cv2.destroyAllWindows()
        
        
        if len(refPt) < 7:
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f'Point ({x},{y}) saved')
                refPt.append([x,y])
                cv2.imshow("image", img)
                
    
    #Set windows size
    
    cv2.imshow("image", img)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1920, 1080)
    
    #Call function
    cv2.setMouseCallback("image", click_event)
    
    
    cv2.waitKey(0) 
    print(refPt)
    
    
    ship_dir = rotation(refPt[0], refPt[1])
    turbulent_wake_dir = rotation(refPt[0], refPt[2])
    left_kelvin_wake_dir = rotation(refPt[3], refPt[4])
    right_kelvin_wake_dir = rotation(refPt[5], refPt[6])
    
    # kelvin_angle = abs(ship_dir - kelvin_wake_dir)
    
    turb_angle_left = abs(turbulent_wake_dir - left_kelvin_wake_dir)
    turb_angle_right = abs(turbulent_wake_dir - right_kelvin_wake_dir)
    asymmetry = turb_angle_left - turb_angle_right

    angles.append([file, turb_angle_left, turb_angle_right, asymmetry, turbulent_wake_dir])

    # sys.exit()

#%% Save data
    
with open(r'C:/Users/Ruben/Documents/Thesis/Data/Angles2/Results/sym/symangles5.p', 'wb') as f:    
    pickle.dump(angles, f)
