import jupyter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import sklearn
import tensorflow
import time
import math
import random
import pickle
import time
import tensorflow as tf

"""Banging the data into usable format"""
def randomize_set(x,y):
    numel = len(y)
    print(type(x), type(y))
    print(x.shape, y.shape)
    listicle = [[i] for i in range(numel)]
    random.shuffle(listicle)
    x_shape = x.shape
    y_shape = y.shape
    ret_x = np.ones((x_shape[0],x_shape[1],x_shape[2]))
    ret_y = np.ones((x_shape[0]))
    print(ret_x.shape, ret_y.shape)
    for i in range(numel):
        index = listicle[i]
        curr_x = x[index,:,:]
        curr_y = y[index]
        ret_x[i,:,:] = curr_x
        ret_y[i] = curr_y
        #print(index)
    return(ret_x,ret_y)

def make_one_hot_encoding(y, num_labels):
    print('Making one hot encoding')
    print(type(y), type(num_labels))
    print(y.shape, num_labels)
    y_shape = y.shape
    numel = y_shape[0]
    print(numel)
    #for i in range(numel):
    ret_y = np.zeros((numel, num_labels))
    print('Return y = ', ret_y.shape)
    for i in range(numel):
        curr_label = y[i]
        #print(i, curr_label)
        curr_encoding = np.zeros(num_labels)
        for j in range(num_labels):
            if j == int(curr_label):
                #print('Match!', j, curr_label)
                curr_encoding[j] = 1.0
        #print(curr_encoding)
        ret_y[i] = curr_encoding
    return ret_y

def expand_x(x):
    shape_x = x.shape
    print('Length is = ', len(shape_x))
    if len(shape_x) == 3:
        print('Expanding')
        ret_x = np.empty((shape_x[0],shape_x[1],shape_x[2],1))
        ret_x[:,:,:,0] = x
        print(ret_x.shape)
        print('Example value = ', ret_x[0,0,0,0])
    return(ret_x)

"""Neural Network helper functions"""
def conv2d(x, W, b, strides=1):
	"""
	Args:
		x
		W
		b
		strides
	Returns:
		TBD
	"""
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)

def maxpool2d(x, k=2):
	"""
	Args:
		x
		k
	Returns:
		TBD
	"""
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

# Create model
def conv_net(x, weights, biases):
	"""
	Args:
  		x:
  		weights:
  		biases:

	Returns:
		out:

	"""
    # Layer 1
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    # Layer 2
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    # Layer 3
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv2)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(
        conv3,
        [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(
        tf.matmul(fc1, weights['fully_connected']),
        biases['fully_connected'])
    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
