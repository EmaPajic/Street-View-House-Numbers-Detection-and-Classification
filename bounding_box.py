#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:56:52 2020
@author: EmaPajic
"""

import sys
sys.path.insert(0, '/home/user/miniconda3/envs/myenv/lib/python3.5/site-packages')
import cv2
import numpy as np


class BoundingBox:
    def __init__(self, center_x, center_y, width, height,
                 class_scores = None, confidence = None):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        
        self.class_scores = class_scores
        self.confidence = confidence
    
    def get_label(self):  
        if self.class_scores == None:
            return None
        return np.argmax(self.class_scores)
    
    def get_score(self):
        label = self.get_label()
        if label == None:
            return 0
        return self.class_scores[label]
    
    def get_box_as_array(self):
        return np.array([self.center_x, self.center_y,
                         self.width, self.height])

    def intersection_over_union(self, bounding_box):
        return iou(self, bounding_box)
    
def to_edge_box(center_box):
    start_x = center_box.center_x - center_box.width / 2
    end_x = center_box.center_x + center_box.width / 2
    start_y = center_box.center_y - center_box.height / 2
    end_y = center_box.center_y + center_box.height / 2
    return start_x, end_x, start_y, end_y
    
def to_center_box(edge_box):
    start_x, end_x, start_y, end_y = edge_box
    center_x = (start_x + end_x) / 2
    center_y = (start_y + end_y) / 2
    width = end_x - start_x
    height = end_y - start_y
    return center_x, center_y, width, height

def iou(box1, box2):
    def _interval_overlap(interval_a, interval_b):
            start_a, end_a = interval_a
            start_b, end_b = interval_b
            
            if start_b < start_a:
                if end_b < start_a:
                    return 0
                else:
                    return min(end_a, end_b) - start_a
            else:
                if end_a < start_b:
                    return 0
                else:
                    return min(end_a, end_b) - start_b
               
    start_x1, end_x1, start_y1, end_y1 = to_edge_box(box1)
    start_x2, end_x2, start_y2, end_y2 = to_edge_box(box2)

    intersect_w = _interval_overlap([start_x1, end_x1], [start_x2, end_x2])
    intersect_h = _interval_overlap([start_y1, end_y1], [start_y2, end_y2])
    intersection = intersect_w * intersect_h
    union = box1.width * box1.height + \
            box2.width * box2.height - intersection
    
    return float(intersection) / union

def draw_boxes(image, boxes):
    for box in boxes:
        start_x, end_x, start_y, end_y = to_edge_box(box)
        #cv2.imshow('image fun', image)
        cv2.rectangle(image, (int(start_x), int(start_y)),
                      (int(end_x), int(end_y)), (0, 255, 0), 2)
        label = box.get_label()
        score = box.get_score()
        cv2.putText(image,
                '{}: {:.2f}'.format(label, score),
                (int(start_x), int(start_y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1e-3 * image.shape[0],
                (0, 255, 0), 1)
    return image

