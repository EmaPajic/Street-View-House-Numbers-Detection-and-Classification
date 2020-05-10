#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: EmaPajic
"""

import numpy as np
import os
import cv2
from colors import get_color

class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c = c
        self.classes = classes
        
        self.label = -1
        self.score = -1
        
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
            
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def _interval_overlap(interval_a, interval_b):
    a_start, a_end = interval_a
    b_start, b_end = interval_b
    
    if b_start < a_start:
        if b_end < a_start:
            return 0
        else:
            return min(a_end, b_end) - a_start
    else:
        if a_end < b_start:
            return 0
        else:
            return min(a_end, b_end) - b_start

def bbox_iou(box1, box2):
    width_overlap = _interval_overlap([box1.xmin, box1.xmax],
                                      [box2.xmin, box2.xmax])
    height_overlap = _interval_overlap([box1.ymin, box1.ymax],
                                       [box2.ymin, box2.yman])
    
    intersection = width_overlap * height_overlap
    
    area_box1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    area_box2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    
    union = area_box1 + area_box2 - intersection
    
    return float(intersection) / union

def draw_boxes(image, boxes, labels, obj_treshold):
    for box in boxes:
        label_str = ''
        
        label = box.get_label()
        score = box.get_score()
        
        if score > obj_treshold:
            label_str += (label + ' ' + str(round(score*100, 2)) + '%')
        else:
            label = -1
        
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX,
                                        1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin),
                          pt2=(box.xmax,box.ymax),
                          color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)         
    
    return image

def non_max_supression(boxes, nms_treshold):
    if len(boxes) == 0:
        return
    number_of_classes = len(boxes[0].classes)
    
    for c in range(number_of_classes):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        
        for i in range(len(sorted_indices)):
            for j in range(i + 1, len(sorted_indices)):
                if bbox_iou(boxes[sorted_indices[i]],
                            boxes[sorted_indices[j]] >= nms_treshold):
                    boxes[sorted_indices[j]].classes[c] = 0