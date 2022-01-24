#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 21:50:21 2021

@author: a8hik
"""
from os import listdir
from os.path import isfile, join 
import os
import numpy as np
import cv2 as cv
import skimage.transform 
from PIL import Image
from cityscapesscripts.helpers import labels

MEAN = (71.60167789, 82.09696889, 72.30508881)
CLASSES = 20

HEIGHT = 512
WIDTH = 1024

def sub_mean_chw(data):
   data = data.transpose((1, 2, 0))  # CHW -> HWC
   data -= np.array(MEAN)  # Broadcast subtract
   data = data.transpose((2, 0, 1))  # HWC -> CHW
   return data

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image

def color_map(output):
   output = output.reshape(CLASSES, HEIGHT, WIDTH)
   out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
   for x in range(WIDTH):
       for y in range(HEIGHT):
           if (np.argmax(output[:, y, x] )== 19):
               out_col[y,x] = (0, 0, 0)
           else:
               out_col[y, x] = labels.id2label[labels.trainId2label[np.argmax(output[:, y, x])].id].color
   return out_col

def file_names(data_path):
    dir_map = {i:f for i,f in enumerate(listdir(data_path))} 
    return dir_map

def file_names_dir(data_path):
    data_dir = join(data_path, listdir(data_path)[0])
    dir_map = {i:join(data_path,f) for i,f in enumerate(listdir(data_path))} 
    return dir_map   

def return_processed(data_path, start, batch_size):
    data_dir = join(data_path, [f for f in listdir(data_path) if f != 'out'][0])
    dir_map = file_names_dir(data_dir)
    print(dir_map)
    data = [sub_mean_chw(np.array(rescale_image(np.asarray(Image.open(dir_map[i])), (512, 1024),order=1)).transpose((2, 0, 1))) for i in range(start, start+batch_size)]
    return np.array(data)

