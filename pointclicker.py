# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:04:17 2021

@author: Ruben
"""


import cv2
import numpy as np

imagefolder = r'C:\Users\Ruben\Documents\Thesis\Data\Angles1\Images'
imagepath = r'C:\Users\Ruben\Documents\Thesis\Data\Angles1\Images\Schip8filtered.png'



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


def rotation(p1, p2):
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    rc = delta_y / delta_x
    ang = np.arctan(rc) * (180 / np.pi)
    return ang


ship_dir = rotation(refPt[0], refPt[1])
turbulent_wake_dir = rotation(refPt[0], refPt[2])
kelvin_wake_dir = rotation(refPt[3], refPt[4])
kelvin_angle = abs(ship_dir - kelvin_wake_dir)
turb = abs(turbulent_wake_dir - kelvin_wake_dir)

print(f'Kelvin angle with respect to ship is {kelvin_angle}, with respect to the turbulent wake is {turb}')
