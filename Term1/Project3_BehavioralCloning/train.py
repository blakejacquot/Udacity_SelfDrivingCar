import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os
import cv2
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
import json
import os
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.core import Flatten,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, model_from_json
np.random.seed(1337)  # for reproducibility
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU

import time
import sys


def examine_data(i, y, X):
    """ Display details of chosen label and image data
    """
    print('Label = ', y[i], type(y[i]))
    print(np.max(X), np.min(X))
    #image = X[i,:,:,:]
    #print('Shape of image = ', image.shape)
    print(np.min(X), np.max(X))
    im1 = X[i+0,:,:,0]
    im2 = X[i+1,:,:,0]
    im3 = X[i+2,:,:,0]
    print('Angle extremes = ', np.min(y), np.max(y))
    print('Image shapes = ', im1.shape, im2.shape, im3.shape)
    print('Steering angles = ', y[i+0], y[i+1], y[i+2])
    min = np.min(im1)
    max = np.max(im2)
    print(min, max)
    plt.figure
    plt.subplot(131)
    plt.imshow(im1, cmap='gray')
    plt.title('im1')
    plt.subplot(132)
    plt.imshow(im2, cmap='gray')
    plt.title('im2')
    plt.subplot(133)
    plt.imshow(im3, cmap='gray')
    plt.title('im3')
    plt.show()

def load_pickle(file):
    print('Loading dataset from file')
    data = pickle.load(open(file, "rb" ))
    y = data['y']
    X = data['X']
    return y, X

def write_pickle_file(filename, y, X):
    print('Saving to pickle file')
    data_to_save = {'y': y,
                    'X': X,
                    }
    pickle.dump(data_to_save, open(filename, "wb" ))

def data_generator(X, y, batch_size):
    inputs = []
    targets = []
    size = 0
    X_shape = X.shape
    y_shape = y.shape
    inputs = np.zeros((batch_size, X_shape[1], X_shape[2], 1))
    targets = np.zeros((batch_size))
    while True:
        for i in range(len(y)):
            image = X[i,:,:,:]
            steer_angle = y[i]
            inputs[size,:,:,:] = image
            targets[size] = steer_angle
            size += 1
            if size == batch_size:
                inputs = inputs.astype('float32')
                targets = targets.astype('float32')
                return_tuple = (np.array(inputs),np.array(targets))
                yield (return_tuple)
                size = 0
                inputs = np.zeros((batch_size, X_shape[1], X_shape[2], 1))
                targets = np.zeros((batch_size))

def build_model(input_shape, num_labels):

    """
    9 layers (1 normalization, 5 convnet, 3 FC)

    Conv1-3, 2x2 stride, 5x5 kernel
    Conv4-5, no stride, 3x3 kernel

    """

    pool_size = (2, 2)

    model = Sequential()
    """Convnet layer 1"""
    model.add(Convolution2D(24,5,5, border_mode='valid', subsample=(2,2), input_shape=input_shape))
    #model.add(Convolution2D(24,5,5, activation='relu', border_mode='valid', subsample=(2,2), input_shape=input_shape))
    #model.add(Convolution2D(24,5,5,border_mode='valid', input_shape=input_shape))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.1))
    #model.add(ELU())

    #model.add(Dropout(0.1))
    """Convnet layer 2"""
    model.add(Convolution2D(36,5,5, border_mode='valid', subsample=(2,2)))
    #model.add(Convolution2D(36,5,5, activation='relu', border_mode='valid', subsample=(2,2), input_shape=input_shape))
    #model.add(Convolution2D(36,5,5,border_mode='valid'))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.1))
    #model.add(ELU())

    """Convnet layer 3"""
    model.add(Convolution2D(48,5,5, border_mode='valid', subsample=(2,2)))
    #model.add(Convolution2D(48,5,5, activation='relu', border_mode='valid', subsample=(2,2), input_shape=input_shape))
    #model.add(Convolution2D(48,5,5,border_mode='valid'))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    #model.add(Activation('relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.1))
    #model.add(ELU())

    """Convnet layer 4"""
    model.add(Convolution2D(64,3,3, border_mode='valid', subsample=(1,1)))
    #model.add(Convolution2D(64,3,3, activation='relu', border_mode='valid', subsample=(1,1), input_shape=input_shape))
    #model.add(Convolution2D(64,3,3,border_mode='valid'))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.1))
    #model.add(ELU())

    model.summary()

    """Convnet layer 5"""
    model.add(Convolution2D(64,5,1, border_mode='valid', subsample=(1,1)))
    #model.add(Convolution2D(64,3,3, activation='relu', border_mode='valid', subsample=(1,1), input_shape=input_shape))
    #model.add(Convolution2D(64,3,3,border_mode='valid'))
    #model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.1))
    #model.add(ELU())

    #model.add(Dropout(0.1))
    """Flatten"""
    model.add(Flatten())

    """First fully-connected layer"""
    model.add(Dense(100))
    #model.add(Activation('relu'))
    """Second fully-connected layer"""
    model.add(Dense(50))
    #model.add(Activation('relu'))
    """Third fully-connected layer"""
    model.add(Dense(10))
    #model.add(Activation('relu'))
    """Output layer"""
    model.add(Dense(num_labels))
    return model

def load_model():
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))
    print('File open')
    model.load_weights('model.h5')
    return model

def train_on_path(data_path_root, batch_size, nb_epoch, learning_rate):
    train_path = os.path.join(data_path_root, 'data_train.p')
    test_path = os.path.join(data_path_root, 'data_test.p')
    val_path = os.path.join(data_path_root, 'data_val.p')
    num_labels = 1 # the output is a single steering angle

    # Load data
    y_train, X_train = load_pickle(train_path)
    y_test, X_test = load_pickle(test_path)
    y_val, X_val = load_pickle(val_path)

    # Ensure everything is float32
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')

    # Report info on label and data shape
    print('Train shape y, X', y_train.shape, X_train.shape)
    print('Test shape y, X', y_test.shape, X_test.shape)
    print('Val shape y, X', y_val.shape, X_val.shape)

    # Define input shape
    X_shape = X_test.shape
    print('X test shape = ', X_shape[1], X_shape[2], X_shape[3])
    input_shape = X_test.shape[1:]
    print('Input shape = ', input_shape)

    curr_dir_contents = os.listdir()

    if 'model.json' in curr_dir_contents:
        print('Loading model for continued training')
        model = load_model()
    else:
        print('Model does not exist. Making it for the first time')
        model = build_model(input_shape, num_labels)

    #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam',loss='mse')
    model.summary()


    """Train model"""
    print('Training model')
    num_train_points = len(y_train)
    train_samples_per_epoch = batch_size * math.floor(num_train_points/batch_size)
    generator = data_generator(X_train, y_train, batch_size) # a tuple of (inputs, targets)
    print('num_train_points = ', num_train_points)
    print('train_samples_per_epoch  = ', train_samples_per_epoch)

    #examine_data(1, y_train, X_train)

    history = model.fit_generator(generator, train_samples_per_epoch, nb_epoch, verbose=1,
                        validation_data=(X_val, y_val))

    """Save model"""
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
        # save weights
        model.save_weights('./model.h5')
        print("Model saved")

    """Evaluate model"""
    print('Evaluating model')
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss = ', score)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])

if __name__ == '__main__':

    batch_size = 32
    nb_epoch = 10
    learning_rate = 0.001
    # User-defined variables
    # User-defined variables
    data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/Data_from_Udacity/'
    train_on_path(data_path_root, batch_size, nb_epoch, learning_rate)
    #data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset1/'
    #train_on_path(data_path_root, batch_size, nb_epoch, learning_rate)
    #data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset2/'
    #train_on_path(data_path_root, batch_size, nb_epoch, learning_rate)
    #data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset3/'
    #train_on_path(data_path_root, batch_size, nb_epoch, learning_rate)



