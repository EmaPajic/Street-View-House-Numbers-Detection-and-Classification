#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:10:30 2020
@author: EmaPajic
"""

import sys
sys.path.insert(0, '/home/user/miniconda3/envs/myenv/lib/python3.5/site-packages')
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def read_dataset(img_folder):
    img_file_names = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    images = [] 
    for img_name in img_file_names:
        fname = os.path.join(img_folder, img_name)
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        #plt.figure()
        #plt.imshow(img)
        #plt.show()
    return images
    

if __name__ == '__main__':
    img_folder = '/home/user/Desktop/4.god/Neuralne/projekat/probni/'
    read_dataset(img_folder)


