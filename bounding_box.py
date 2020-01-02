#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:56:52 2020
@author: EmaPajic
"""

sys.path.insert(0, '/home/user/miniconda3/envs/myenv/lib/python3.5/site-packages')
import cv2
import numpy as np

class BoundingBox:
    def __init__(self, x, y, width, height,
                 class_scores = None, confidence = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        self.class_scores = class_scores
        self.confidence = confidence
    
    def get_label(self):
        return np.argmax(self.class_scores)
    
    def get_score(self):
        return self.class_scores[self.get_label()]
    
    def get_box_as_array(self):
        return np.array([self.x, self.y, self.width, self.height])
    
    def intersection_over_union(self, bounding_box):
        return 1