#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge 



def detect (image : Image):
    img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    bridge = CvBridge()
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
    # blank1 = np.zeros(img.shape, dtype= 'uint8')
    # for contour in contours:
    #     mask = np.zeros_like(gray)
    #     cv2.drawContours(mask, [contour], 0, (255), -1)
    #     cv2.fillPoly(blank1, [contour], (0, 0, 255))

    # cv2.imshow('result',blank1)
    # image = bridge.cv2_to_imgmsg(img, encoding='bgr8')    
    # pub1.publish(image)
    #pub.publish(image)
    

    cv2.waitKey(1)
    

if __name__ == '__main__' :
    
    rospy.init_node("Object_detect")
    pub1=rospy.Publisher("/camera/color/object_detection", Image, queue_size=10)
    #pub=rospy.Publisher("/camera/color/image_raw", Image, queue_size=10)
    sub = rospy.Subscriber("/zed2i/zed_node/rgb_raw/image_raw_color", Image, callback = detect)
    
    rospy.spin()
