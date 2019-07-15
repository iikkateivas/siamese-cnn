# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:20:38 2019

@author: Iikka
"""

import re
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import os
import cv2


size = 2
total_sample_size = 4000

from PIL import Image
import glob
image_list = []

path = os.getcwd()
for f in glob.glob('./gen_data_train\s1' + '/*.jpeg'): #assuming gif
    image_list.append(f)

path = os.getcwd()

def get_data(size, total_sample_size):
    
    image_list = []

    for f in glob.glob('./gen_data_train\s1' + '/*.jpeg'): #assuming gif
        image_list.append(f)
    #read the image
    image = cv2.imread(image_list[0],-1)
    #reduce the size
    image = image[::size, ::size]
    #get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0
    
    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2]) # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])
    
    for i in range(34):
        for j in range(int(total_sample_size/34)):
            ind1 = 0
            ind2 = 0
            
            #read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)
            
            image_list = []
            for f in glob.glob('gen_data_train\s' + str(i+1) + '/*.jpeg'):
                image_list.append(f)
            
            
            # read the two images
            img1 = cv2.imread(image_list[ind1],-1)
            img2 = cv2.imread(image_list[ind2],-1)
            
            #reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            
            #store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
            
            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 0
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    
    for i in range(int(total_sample_size/10)):
        for j in range(10):
            
            #read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(34)
                ind2 = np.random.randint(34)
                if ind1 != ind2:
                    break
                
            image_list_1 = []
            for f in glob.glob('gen_data_train\s' + str(ind1+1) + '/*.jpeg'): #assuming gif
                image_list_1.append(f)
                
            image_list_2 = []
            for f in glob.glob('gen_data_train\s' + str(ind2+1) + '/*.jpeg'): #assuming gif
                image_list_2.append(f)   
                    
            img1 = cv2.imread(image_list_1[j],-1)
            img2 = cv2.imread(image_list_2[j],-1)

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 1
            count += 1
            
    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y

X, Y = get_data(size, total_sample_size)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)


def build_base_network(input_shape):
    
    seq = Sequential()
    
    nb_filter = [6, 12]
    kernel_size = 3
    
    
    #convolutional layer 1
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2))) 
    seq.add(Dropout(.25))
    
    #convolutional layer 2
    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th')) 
    seq.add(Dropout(.25))

    #flatten 
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq

input_dim = x_train.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

#def euclidean_distance(vects):
#    x, y = vects
#    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

epochs = 20
rms = RMSprop()

model = Model(input=[img_a, img_b], output=distance)

#def contrastive_loss(y_true, y_pred):
#    margin = 1
#    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1-y_true) * sqaure_pred + y_true * margin_square)

model.compile(loss=contrastive_loss, optimizer=rms)

img_1 = x_train[:, 0]
img_2 = x_train[:, 1] 

model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=2, nb_epoch=epochs)

pred = model.predict([x_test[:, 0], x_test[:, 1]])

pred2 = np.rint(pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred2)


i = 37
j = 37
ind1 = 0
ind2 = 7
size = 2

image_list_1 = []
for f in glob.glob('gen_data_test\s' + str(i) + '/*.jpeg'): #assuming gif
    image_list_1.append(f)
    
image_list_2 = []
for f in glob.glob('gen_data_test\s' + str(j) + '/*.jpeg'): #assuming gif
    image_list_2.append(f)   
        
img1 = cv2.imread(image_list_1[ind1],-1)
img2 = cv2.imread(image_list_2[ind2],-1)

#reduce the size
img1 = img1[::size, ::size]/255
img2 = img2[::size, ::size]/255


x_1 = np.zeros([1, 1, 56, 46])
x_2 = np.zeros([1, 1, 56, 46])

x_1[0, 0, :, :] = img1
x_2[0, 0, :, :] = img2

pred = model.predict([x_1, x_2])
print(pred)

import time
t = time.time()
pred = model.predict([x_1, x_2])
elapsed = time.time() - t


i = 35
j = 40
ind1 = 1
ind2 = 2
size = 2

img1 = cv2.imread('test_data2/s' + str(i) + '/' + str(ind1) + '.pgm',-1)
img2 = cv2.imread('test_data2/s' + str(j) + '/' + str(ind2) + '.pgm',-1)

#reduce the size
img1 = img1[::size, ::size]/255
img2 = img2[::size, ::size]/255


x_1 = np.zeros([1, 1, 56, 46])
x_2 = np.zeros([1, 1, 56, 46])

x_1[0, 0, :, :] = img1
x_2[0, 0, :, :] = img2

pred = model.predict([x_1, x_2])
print(pred)
