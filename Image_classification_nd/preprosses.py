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

base_dir = 'data/raw-img'
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo", "ragno": "spider"}


def return_dataset_with_labels():
    classes = [nf for nf in listdir(base_dir)]
    data_set = [] 
    for nf in classes[:1]:
        new_dir = join(base_dir, nf)
        data = [(np.array(cv.resize(cv.imread(join(new_dir,f)), (224,224)), dtype=np.float32, order='C'), translate[nf]) for f in listdir(new_dir)]
        data_set.append(data)
    return np.array(data_set)

def return_dataset():
    classes = [nf for nf in listdir(base_dir)]
    data_set = [] 
    for nf in classes[:1]:
        new_dir = join(base_dir, nf)
        data = np.array([cv.resize(cv.imread(join(new_dir,f)), (224,224)) for f in listdir(new_dir)])
        data_set.append(data)
    return np.array(data_set)


#folder_directories = os.chdir()


