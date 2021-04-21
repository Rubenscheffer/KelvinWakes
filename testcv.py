# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:04:17 2021

@author: Ruben
"""


import cv2

imagepath = r'C:\Users\Ruben\Documents\Thesis\Data\Angles1\Images\Schip2filtered.png'



#This variable we use to store the pixel location
refPt = []
img = cv2.imread(imagepath)



#click event function
def click_event(event, x, y, flags, param):
    if len(refPt) < 5:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,",",y)
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x)+", "+str(y)
            refPt.append([x,y])
            cv2.imshow("image", img)


#standard window

cv2.imshow("image", img)

#
cv2.setMouseCallback("image", click_event)


cv2.waitKey(0) 
print(refPt)
cv2.destroyAllWindows()

