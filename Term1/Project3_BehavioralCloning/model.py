"""
Model for Udacity self-driving car class project 3
"""

import json
import math
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten

np.random.seed(1337)  # for reproducibility


def examine_data(i, y_labels, x_data):
    """ Display details of chosen label and image data
    """
    print('Label = ', y_labels[i], type(y_labels[i]))
    print(np.max(x_data), np.min(x_data))
    print(np.min(x_data), np.max(x_data))
    im1 = x_data[i+0, :, :, 0]
    im2 = x_data[i+1, :, :, 0]
    im3 = x_data[i+2, :, :, 0]
    print('Angle extremes = ', np.min(y_labels), np.max(y_labels))
    print('Image shapes = ', im1.shape, im2.shape, im3.shape)
    print('Steering angles = ', y_labels[i+0], y_labels[i+1], y_labels[i+2])
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
    """ Load pickle file.
    """
    print('Loading dataset from file')
    data = pickle.load(open(file, "rb"))
    print(data.keys())
    y_labels = data['y']
    x_data = data['X']
    return y_labels, x_data

def write_pickle_file(filename, y_labels, x_data):
    """ Write pickle file
    """
    print('Saving to pickle file')
    data_to_save = {'y': y_labels,
                    'X': x_data}
    pickle.dump(data_to_save, open(filename, "wb"))

def data_generator(x_data, y_labels, batch_size):
    """ Make data generator.
    """
    inputs = []
    targets = []
    size = 0
    x_shape = x_data.shape
#    y_shape = x_data.shape
    inputs = np.zeros((batch_size, x_shape[1], x_shape[2], 1))
    targets = np.zeros((batch_size))
    while True:
        for i in range(len(y_labels)):
            image = x_data[i, :, :, :]
            steer_angle = y_labels[i]
            inputs[size, :, :, :] = image
            targets[size] = steer_angle
            size += 1
            if size == batch_size:
                inputs = inputs.astype('float32')
                targets = targets.astype('float32')
                return_tuple = (np.array(inputs), np.array(targets))
                yield return_tuple
                size = 0
                inputs = np.zeros((batch_size, x_shape[1], x_shape[2], 1))
                targets = np.zeros((batch_size))

def build_model(input_shape, num_labels):
    """
    9 layers (1 normalization, 5 convnet, 3 FC)

    Conv1-3, 2x2 stride, 5x5 kernel
    Conv4-5, no stride, 3x3 kernel
    """

    model = Sequential()

    # Convnet layer 1
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2),
                            input_shape=input_shape))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Convnet layer 2
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(ELU())

    # Convnet layer 3
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Convnet layer 4
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.summary()

    # Convnet layer 5
    model.add(Convolution2D(64, 3, 1, border_mode='valid', subsample=(1, 1)))
    #model.add(MaxPooling2D())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Flatten
    model.add(Flatten())

    # First fully-connected layer
    model.add(Dense(100))

    # Second fully-connected layer
    model.add(Dense(50))

    # Third fully-connected layer
    model.add(Dense(10))

    # Output layer
    model.add(Dense(num_labels))

    return model

def load_model():
    """ Load existing trained model.
    """
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))
    print('File open')
    model.load_weights('model.h5')
    return model

def train_on_path(data_path_root, batch_size, nb_epoch, learning_rate):
    """ Use a data path for training
    """
    train_path = os.path.join(data_path_root, 'data_train.p')
    test_path = os.path.join(data_path_root, 'data_test.p')
    val_path = os.path.join(data_path_root, 'data_val.p')
    num_labels = 1 # the output is a single steering angle

    # Load data
    y_train, x_train = load_pickle(train_path)
    y_test, x_test = load_pickle(test_path)
    y_val, x_val = load_pickle(val_path)

    # Ensure everything is float32
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')

    # Report info on label and data shape
    print('Train shape y, x', y_train.shape, x_train.shape)
    print('Test shape y, x', y_test.shape, x_test.shape)
    print('Val shape y, x', y_val.shape, x_val.shape)

    # Define input shape
    x_shape = x_test.shape
    print('x test shape = ', x_shape[1], x_shape[2], x_shape[3])
    input_shape = x_test.shape[1:]
    print('Input shape = ', input_shape)

    curr_dir_contents = os.listdir()

    if 'model.json' in curr_dir_contents:
        print('Loading model for continued training')
        model = load_model()
    else:
        print('Model does not exist. Making it for the first time')
        model = build_model(input_shape, num_labels)

    Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train model
    print('Training model')
    num_train_points = len(y_train)
    train_samples_per_epoch = batch_size * math.floor(num_train_points/batch_size)
    generator = data_generator(x_train, y_train, batch_size) # a tuple of (inputs, targets)
    print('num_train_points = ', num_train_points)
    print('train_samples_per_epoch  = ', train_samples_per_epoch)

    #examine_data(1, y_train, x_train) # Used for trouble-shooting

    history = model.fit_generator(generator, train_samples_per_epoch, nb_epoch, verbose=1,
                                  validation_data=(x_val, y_val))
    # Save model
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
        model.save_weights('./model.h5') # save weights
        print("Model saved")

    # Evaluate model
    print('Evaluating model')
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss = ', score)

if __name__ == '__main__':
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/Data_from_Udacity/'
#     train_on_path(DATA_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE)
#
#     # Basic driving
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset1/'
#     train_on_path(DATA_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE)
#
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset2/'
#     train_on_path(DATA_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE)
#
#     # Super drunk driving dataset. Use to recover from extreme situations
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset3/'
#     train_on_path(DATA_PATH, BATCH_SIZE, 5, LEARNING_RATE)
#
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset4/'
#     train_on_path(DATA_PATH, BATCH_SIZE, 30, LEARNING_RATE)

#
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/Data_from_Udacity/'
#     train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)
#
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset1/'
#     train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)
#
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset2/'
#     train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)
#
#     # Super drunk driving dataset. Use to recover from extreme situations
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset3/'
#     train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)
#
#     DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset4/'
#     train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)



    DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/Data_from_Udacity/'
    train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)

    DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset1/'
    train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)

#    DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset2/'
#    train_on_path(DATA_PATH, BATCH_SIZE, 5, 0.0001)

#    # Super drunk driving dataset. Use to recover from extreme situations
#    DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset3/'
#    train_on_path(DATA_PATH, 64, 10, 0.0001)

#    DATA_PATH = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset4/'
#    train_on_path(DATA_PATH, 64, 10, 0.0001)



