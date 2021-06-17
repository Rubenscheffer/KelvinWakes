# -*- coding: utf-8 -*-
"""
For instructions read the readme

"""

import cv2
import numpy as np
import os
import pickle
import sys

imagefolder = r'C:\Users\Ruben\Documents\Thesis\Data\Angles2\Images'

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
        if len(refPt) >= 5:
            cv2.destroyAllWindows()
        
        
        if len(refPt) < 5:
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
    kelvin_wake_dir = rotation(refPt[3], refPt[4])
    kelvin_angle = abs(ship_dir - kelvin_wake_dir)
    turb_angle = abs(turbulent_wake_dir - kelvin_wake_dir)
    
    print(f'Kelvin angle with respect to ship is {kelvin_angle}, with respect to the turbulent wake is {turb_angle}')
    print(f'Direction of ship is {ship_dir}')
    angles.append([file, kelvin_angle, turb_angle, ship_dir, turbulent_wake_dir])

    # sys.exit()

#%% Save data
    
# with open(r'C:/Users/Ruben/Documents/Thesis/Data/Angles2/Results/res/anglesheading5.p', 'wb') as f:    
#     pickle.dump(angles, f)
