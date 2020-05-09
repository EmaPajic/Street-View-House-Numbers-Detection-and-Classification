#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: EmaPajic
"""

import numpy as np
import os
import cv2
from bounding_boxes import BoundingBox, bbox_iou
from scipy.special import expit, softmax

def _sigmoid(x):
    return expit(x)

def _softmax(x):
    return softmax(x)

def normalize(image):
    return image/255.

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
            
def preprocess_input(image, network_height, network_width):
    new_height, new_width, _ = image.shape
    
    if (float(network_width)/new_width < float(network_height)/new_height):
        new_height = (new_height * network_width) // new_width
        new_width = network_width
    else:
        new_width = (new_width * network_height) // new_height
        new_height = network_height
        
    resized_image = cv2.resize(image[:,:,::-1]/255., (new_width, new_height))

def get_yolo_boxes():
    pass

def correct_yolo_boxes(boxes, image_height, image_width,
                       network_heigh, network_width):
    pass

def evaluate():
    pass

def compute_average_precision(recall, precision):
    pass