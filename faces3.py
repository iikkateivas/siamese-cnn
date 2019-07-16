# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:20:38 2019

@author: Iikka
"""

import numpy as np
import cv2
import glob


from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


size = 2
total_sample_size = 7000

# generate pairs
# Input: size reduction multiplier, total sample size, class amount, samples per class 
def get_data(size, total_sample_size, classes, samples):
    
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
    
    # initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_positive_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2]) # 2 is for pairs
    y_positive = np.zeros([total_sample_size, 1])
    
    for i in range(classes):
        for j in range(int(total_sample_size/classes)):
            ind1 = 0
            ind2 = 0
            
            # read images from same directory (positive pair)
            while ind1 == ind2:
                ind1 = np.random.randint(samples)
                ind2 = np.random.randint(samples)
            
            image_list = []
            for f in glob.glob('gen_data_train\s' + str(i+1) + '/*.jpeg'):
                image_list.append(f)
            
            
            # read the two images
            img1 = cv2.imread(image_list[ind1],-1)
            img2 = cv2.imread(image_list[ind2],-1)
            
            # reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            
            # store the images to the initialized numpy array
            x_positive_pair[count, 0, 0, :, :] = img1
            x_positive_pair[count, 1, 0, :, :] = img2
            
            # assign positive pairs as 0 like in the original contrastive loss paper
            y_positive[count] = 0
            count += 1

    count = 0
    x_negative_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_negative = np.zeros([total_sample_size, 1])
    
    for i in range(int(total_sample_size/samples)):
        for j in range(samples):
            
            # read images from different directory (negative pair)
            while True:
                ind1 = np.random.randint(classes)
                ind2 = np.random.randint(classes)
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

            x_negative_pair[count, 0, 0, :, :] = img1
            x_negative_pair[count, 1, 0, :, :] = img2
            
            # assign negative pairs as 1 like in the original contrastive loss paper
            y_negative[count] = 1
            count += 1
            
    # Concat all data
    X = np.concatenate([x_positive_pair, x_negative_pair], axis=0)/255
    Y = np.concatenate([y_positive, y_negative], axis=0)
    return X, Y
    
def build_base_network(input_shape):
    
    seq = Sequential()
    
    seq.add(Convolution2D(6, 3, 3, input_shape=input_shape, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2))) 
    seq.add(Dropout(.25))
    
    seq.add(Convolution2D(12, 3, 3, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th')) 
    seq.add(Dropout(.25))

    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1-y_true) * sqaure_pred + y_true * margin_square)

def build_siamese_network(x_train):  
    input_dim = x_train.shape[2:] # shape of one image (1, 56, 46)
    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)
    
    base_network = build_base_network(input_dim)
    feat_vecs_a = base_network(img_a)
    feat_vecs_b = base_network(img_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
    model = Model(inputs=[img_a, img_b], outputs=distance)
    model.compile(loss=contrastive_loss, optimizer=RMSprop())
    return model

def validate(model):
    ind1 = 2
    ind2 = 1
    size = 2
    for i in range(0, 6):
        res = []
        for j in range(0, 6):
            img1 = cv2.imread('test_data2/s' + str(i+35) + '/' + str(ind1) + '.pgm',-1)
            img2 = cv2.imread('test_data2/s' + str(j+35) + '/' + str(ind2) + '.pgm',-1)
            img1 = img1[::size, ::size]/255
            img2 = img2[::size, ::size]/255
            x_1 = np.zeros([1, 1, 56, 46])
            x_2 = np.zeros([1, 1, 56, 46])      
            x_1[0, 0, :, :] = img1
            x_2[0, 0, :, :] = img2     
            pred = model.predict([x_1, x_2])
            res.append(pred[0][0])
        
    #    print(res)
        if res.index(min(res)) == i:
            print('Correct')
        else:
            print('False')

### main
    
# Generate data pairs
X, Y = get_data(size, total_sample_size, 34, 10)
# Generate train test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

# Build the model
model = build_siamese_network(x_train)

# Split pairs for the model input
img_1 = x_train[:, 0]
img_2 = x_train[:, 1] 

# Train the model
epochs = 40
model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=1, nb_epoch=epochs)

# Predict test set and calc accuracy metric
pred = model.predict([x_test[:, 0], x_test[:, 1]])
accuracy_score(y_test, np.rint(pred))
validate(model)