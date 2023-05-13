#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge 
import math
import struct
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

'''camera_info : 
K: [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [554.254691191187, 0.0, 320.5, -0.0, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]
<origin xyz="0.8 0 1.17" rpy="0 0.2 0" />
'''

def conversion (u,v):
    roll = 0
    pitch = 0
    yaw = 0
    
    
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                     [math.sin(yaw), math.cos(yaw), 0],
                     [0, 0, 1]])
    
    R_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])

    R_roll = np.array([[1, 0, 0],
                     [0, math.cos(roll), -math.sin(roll)],
                     [0, math.sin(roll), math.cos(roll)]])
    
    R = np.matmul((np.matmul(R_yaw, R_pitch)), R_roll)

    a = np.array([[0],
                 [0],
                 [1]])

    nc = np.matmul(R,a)
    nct = np.transpose(nc)

    #K: [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]

    K = np.array([[554.254691191187, 0.0, 320.5],
                 [0.0, 554.254691191187, 240.5],
                 [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)

    uv=np.array([u, v, 1])
    uvt=np.transpose(uv) 

    deno = np.matmul(nct, np.matmul(K_inv, uvt))
    h = 1.17

    const = h/deno

    XYZ = const*np.matmul(K_inv, uvt)

    return(XYZ)

def point_cloud(contours):

    fields = [point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
              point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
              point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
              point_cloud2.PointField('rgba', 12, point_cloud2.PointField.UINT32, 1)]

    header = Header()
    header.frame_id = "zed_camera"

    points = []

    for contour in contours:
        for point in contour:
            u,v = point[0]
            
            coord = conversion(u,v)
            #print(coord)
            X = coord[0]
            Y = coord[1]
            Z = coord[2]
            #print(X, Y, Z)
            r = int(255.0)
            g = int(255.0)
            b = int(255.0)
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [X, Y, Z, rgb]
            points.append(pt)
    
    cloud_msg = point_cloud2.create_cloud(header, fields, points)
    pub.publish(cloud_msg)


            
def detect (image : Image):
    img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT )
    canny = cv2.Canny(img_blur, 125, 175)
    contours, hierarchies = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    point_cloud(contours)
    
  
    

if __name__ == '__main__' :
    
    rospy.init_node("points")
    
    pub=rospy.Publisher("/pointcloud2", PointCloud2, queue_size=10)
    sub = rospy.Subscriber("/camera_ir/vikram/camera/color/image_raw", Image, callback = detect)

    rospy.spin()