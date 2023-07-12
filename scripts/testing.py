#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2 
import numpy as np
from cv_bridge import CvBridge 

# Load the input image
img = cv2.imread('/home/tamoghna/Pictures/test.jpg')
cv2.imshow('original',img)

blank = np.zeros(img.shape, dtype= 'uint8')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY SCALE",gray)
    
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("RGB SCALE",img_rgb)

img_blur = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT )
#cv2.imshow("BLURRED IMAGE",img_blur)

canny = cv2.Canny(img_blur, 125, 175)
cv2.imshow('Canny Edges',canny)
   
blank1 = np.ones(img.shape, dtype='uint8') * 255 
    
contours, hierarchies = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blank1, contours, -1, (0,0,0),-1)
cv2.imshow('CONTOURS DRAWN',blank1)
#print(contours)
for contour in contours:
    #print(contour)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(0,0,0),thickness=-1)
    
cv2.imshow('result',img)


cv2.waitKey(0)


# Wait until 'Ctrl + C' is pressed in the terminal
while True:
    try:
        if cv2.waitKey(1) == 3:
            break
    except KeyboardInterrupt:
        break

cv2.destroyAllWindows()

