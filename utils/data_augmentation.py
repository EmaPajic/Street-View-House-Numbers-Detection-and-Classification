#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: EmaPajic
"""

import numpy as np
import cv2
import copy

def _random_scale(scale):
    scale = np.random.uniform(1, scale)
    if (np.random.randint(2) == 0):
        return scale
    else:
        return 1./scale
    
def _constrain(min_value, max_value, value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value

def random_flip(image, flip):
    if flip == True:
        return cv2.flip(image, 1)
    return image

def random_disort_image(image, hue = 18, saturation = 1.5, exposure = 1.5):
    dhue = np.random(-hue, hue)
    dsaturation = _random_scale(saturation)
    dexposure = _random_scale(exposure)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    
    image[:, :, 1] *= dsaturation
    image[:, :, 2] *= dexposure
    
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] -= (image[:, :, 0] < 0) * 180
    
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOT_HSV2RGB)

def random_scale_and_crop(image, new_width, new_height,
                          network_width, network_height, dx, dy):
    image_resized = cv2.resize(image, (new_width, new_height))
    
    if dx > 0:
        image_resized = np.pad(image_resized, ((0, 0), (dx, 0), (0, 0)),
                               mode = 'constant', constant_values = 127)
    else:
        image_resized = image_resized[:, -dx:, :]
    
    if (new_width + dx) < network_width:
        image_resized = np.pad(image_resized,
                               ((0, 0),
                                (0, network_width - (new_width + dx)),
                                (0, 0)),
                               mode = 'constant', constant_values = 127)
    
    if dy > 0:
        image_resized = np.pad(image_resized, ((dy, 0), (0, 0), (0, 0)),
                               mode = 'constant', constant_values = 127)
    else:
        image_resized = image_resized[-dy:, :, :]
    if (new_height + dy) < network_height:
        image_resized = np.pad(image_resized,
                               ((0, network_height - (new_height + dy)),
                                (0, 0),
                                (0, 0)),
                               mode = 'constant', constant_values = 127)
                               
    return image_resized[:network_height, :network_width, :]
    
def correct_bounding_boxes(boxes, new_width, new_height,
                          network_width, network_height, dx, dy, 
                          flip, image_width, image_height):
    temp_boxes = copy.deepcopy(boxes)
    boxes = []
    np.random.shuffle(temp_boxes)
    
    size_x = float(new_width) / image_width
    size_y = float(new_height) / image_height
    
    for i in range(len(boxes)):
        temp_boxes[i]['xmin'] = int(_constrain(0, network_width, 
             temp_boxes[i]['xmin'] * size_x + dx))
        temp_boxes[i]['xmax'] = int(_constrain(0, network_width, 
             temp_boxes[i]['xmax'] * size_x + dx))
        temp_boxes[i]['ymin'] = int(_constrain(0, network_height, 
             temp_boxes[i]['xmin'] * size_y + dy))
        temp_boxes[i]['ymax'] = int(_constrain(0, network_height, 
             temp_boxes[i]['ymax'] * size_y + dy))
        
            
        if temp_boxes[i]['xmax'] <= temp_boxes[i]['xmin'] or \
        temp_boxes[i]['ymax'] <= temp_boxes[i]['xmax']:
            continue
        
        if flip == True:
            swap = temp_boxes[i]['xmin']
            temp_boxes[i]['xmin'] = network_width - temp_boxes[i]['xmax']
            temp_boxes[i]['xmax'] = network_width - swap
            
        boxes.append(temp_boxes[i])
    return boxes