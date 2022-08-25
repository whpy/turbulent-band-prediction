import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
import os

img = cv2.imread("4010test.png",3)
img_copy = img.copy()
##Convert to greyscale
img_gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
imgray = img_gray
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
image, contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if len(cnt)>50:
        S1 = cv2.contourArea(cnt)
        ell = cv2.fitEllipse(cnt)
        S2 = math.pi*ell[1][0]*ell[1][1]
        if (S1/S2)>0.2:
            img = cv2.ellipse(img,ell,(0,255,0),2)
            print(str(S1)+"  "+str(S2)+str(ell[0][0])+"   "+str(ell[0][1]))
            


##import numpy as np
##import cv2
##
### Read image
##img = cv2.imread('test20.png')
### Smooth it
##img = cv2.medianBlur(img,3)
##img_copy = img.copy()
### Convert to greyscale
##img_gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
### Apply Hough transform to greyscale image
##circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,20,
##                     param1=60,param2=40,minRadius=0,maxRadius=0)
##circles = np.uint16(np.around(circles))
### Draw the circles
##for i in circles[0,:]:
##    # draw the outer circle
##    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
##    # draw the center of the circle
##    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
##cv2.imshow('detected circles',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
