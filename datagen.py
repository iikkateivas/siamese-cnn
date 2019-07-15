# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:26:39 2019

@author: Iikka
"""

import re
import numpy as np
from PIL import Image
import cv2
import os  
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

data_dir = 'gen_data_test'    
source_dir = 'test_data2'

def create_dir(name):
    path = os.getcwd()
    new_dir = path + "/" + name
    try:  
        os.mkdir(new_dir)
    except OSError:  
        print ("Creation of the directory %s failed" % new_dir)
    else:  
        print ("Successfully created the directory %s " % new_dir)
        
create_dir(data_dir)

# define data augmentator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


for i in range(34, 40):
    
    # datagen from the first image
    img = cv2.imread(source_dir + '/s' + str(i+1) + '/' + str(1) + '.pgm',-1)
    x = img_to_array(img)  # array with shape (112, 92, 1)
    x = x.reshape((1,) + x.shape)  # array with shape (1, 112, 92, 1)
    
    dest_dir = data_dir + '/s' + str(i+1)
    create_dir(dest_dir)
    
    j = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=dest_dir, save_prefix='face', save_format='jpeg'):
        j += 1
        if j > 4:
            break  # otherwise the generator would loop indefinitely
    
    # datagen from the second image
    img = cv2.imread(source_dir + '/s' + str(i+1) + '/' + str(2) + '.pgm',-1)
    x = img_to_array(img)  # array with shape (112, 92, 1)
    x = x.reshape((1,) + x.shape)  # array with shape (1, 112, 92, 1)
    
    j = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=dest_dir, save_prefix='cat', save_format='jpeg'):
        j += 1
        if j > 4:
            break  # otherwise the generator would loop indefinitely
    

