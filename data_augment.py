# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:07:48 2020

@author: user
"""

from imgaug import augmenters as iaa
import cv2
import numpy as np

def resize_image_and_boxes(image, boxes, desired_width, desired_height):
    height, width, _ = image.shape
    image = cv2.resize(image, (desired_height, desired_width))
    
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(float(x1) * desired_width / width)
        x2 = int(float(x2) * desired_width / width)
        
        y1 = int(float(y1) * desired_width / width)
        y2 = int(float(y2) * desired_width / width)
        
        new_boxes.append([x1, y1, x2, y2])
    return image, np.array(new_boxes)
        
