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


np.random.seed(1337)  # for reproducibility


def load_pickle(file):
    print('Loading stats from file')
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

if __name__ == '__main__':

    # User-defined variables
    train_path = '/Users/blakejacquot/Desktop/temp2/data_train.p'
    test_path = '/Users/blakejacquot/Desktop/temp2/data_test.p'
    val_path = '/Users/blakejacquot/Desktop/temp2/data_val.p'

    #y_train, X_train = load_pickle(train_path)
    y_test, X_test = load_pickle(test_path)
    y_val, X_val = load_pickle(val_path)


    y_test_shape = y_test.shape
    y_val_shape = y_val.shape
    #print('Train shape y, X', y_train.shape, X_train.shape)
    print('Test shape y, X', y_test.shape, X_test.shape)
    print('Val shape y, X', y_val.shape, X_val.shape)

    print(y_val.shape, X_val.shape)
    print(y_test.shape, X_val.shape)


    #print(y_test)
    print(np.min(y_test), np.max(y_test)) # Steering angle is -1.0 to +1.0



    # TODO: Build a two-layer feedforward neural network with Keras here.
    X_shape = X_test.shape
    print(X_shape[1], X_shape[2], X_shape[3])


    model = Sequential()


    num_labels = 1 # the output is a single steering angle

    input_shape = X_test.shape[1:]
    print(input_shape)

    #first conv layer
    model.add(Convolution2D(24,3,3,border_mode='valid', input_shape=input_shape))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))

    model.add(Convolution2D(36,3,3,border_mode='valid', input_shape=input_shape))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))

    model.add(Convolution2D(48,3,3,border_mode='valid', input_shape=input_shape))
    model.add(MaxPooling2D(dim_ordering='th'))
    model.add(Activation('relu'))

    model.add(Flatten())

    #first fully connected layer
    model.add(Dense(num_labels, activation='relu'))
    #model.add(Dense(128, activation='relu', input_shape=(32*32*3,)))

    model.summary()

    # TODO: Compile and train the model here.
    batch_size = 64
    nb_classes = 1
    nb_epoch = 1

    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

    print('Training model')
    print(X_test.shape, y_test.shape)
    print(X_val.shape, y_val.shape)
    history = model.fit(X_test, y_test,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_val, y_val))
    score = model.evaluate(X_test, y_test, verbose=0)

    print('Evaluating model')
    score = model.evaluate(X_val, y_val, verbose=0) # should be test
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print('Training accuracy = ', history.history['acc'][-1])
    print('Validation accuracy = ', history.history['val_acc'][-1])

# Save the model.
# If the model.json file already exists in the local file,
# warn the user to make sure if user wants to overwrite the model.
if 'model.json' in os.listdir():
	print("The file already exists")
	print("Want to overwite? y or n")
	user_input = input()

	if user_input == "y":
		# Save model as json file
		json_string = model.to_json()

		with open('model.json', 'w') as outfile:
			json.dump(json_string, outfile)

			# save weights
			model.save_weights('./model.h5')
			print("Overwrite Successful")
	else:
		print("the model is not saved")
else:
	# Save model as json file
	json_string = model.to_json()

	with open('model.json', 'w') as outfile:
		json.dump(json_string, outfile)

		# save weights
		model.save_weights('./model.h5')
		print("Saved")



