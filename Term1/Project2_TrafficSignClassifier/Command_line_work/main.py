import jupyter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import sklearn
import tensorflow
print(cv2.__version__)
import time
import math
import random
import pickle
import time
import tensorflow as tf
import sys
import os

import ml_helpers as ml
import pre_proc_helpers as proc

def main():
	# TODO: fill this in based on where you saved the training and testing data
	training_file = '/Users/blakejacquot/Dropbox/MOOCs/Udacity_SelfDrivingCar/Term1/TrafficSignClassifier/traffic-signs-data/train.p'
	testing_file = '/Users/blakejacquot/Dropbox/MOOCs/Udacity_SelfDrivingCar/Term1/TrafficSignClassifier/traffic-signs-data/test.p'
	#checkpoint_path = '/Users/blakejacquot/Desktop/temp2/Udacity_SelfDrivingCar/Term1/Project2_TrafficSignClassifier/Command_line_work/'
	#checkpoint_file = 'model-checkpoint'
	#checkpoint_file_path = os.path.join(checkpoint_path, checkpoint_file)
	restore_model_for_continued_work = 1
	train_model = 0

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)

	X_train, y_train = train['features'], train['labels']
	X_test, y_test = test['features'], test['labels']

	"""Preprocess datasets"""
	X_test_preproc = X_test
	X_test_preproc = proc.make_grayscale(X_test_preproc)
	X_test_preproc = proc.make_gaussian_blur(X_test_preproc, 1)
	X_test_preproc = proc.normalize(X_test_preproc)

	X_train_preproc = X_train
	X_train_preproc = proc.make_grayscale(X_train_preproc)
	X_train_preproc = proc.make_gaussian_blur(X_train_preproc, 1)
	X_train_preproc = proc.normalize(X_train_preproc)

	"""Format training and test data"""
	X_test_gray = proc.make_grayscale(X_test)
	X_train_gray = proc.make_grayscale(X_train)



	"""Randomize the data"""
	[X_test_shuff, y_test_shuff] = ml.randomize_set(X_test_preproc, y_test)
	[X_train_shuff, y_train_shuff] = ml.randomize_set(X_train_preproc, y_train)


	"""One-hot encode the data"""
	y_shuff_onehot_test = ml.make_one_hot_encoding(y_test_shuff, 43)
	y_shuff_onehot_train = ml.make_one_hot_encoding(y_train_shuff, 43)

	"""Expand data dimensions and set up labels"""
	training_data = ml.expand_x(X_train_shuff)
	training_labels = y_shuff_onehot_train
	total_samples = len(training_labels)
	test_data = ml.expand_x(X_test_shuff)
	test_labels = y_shuff_onehot_test

	"""Print data stats"""
	print(X_test_shuff.shape, y_test_shuff.shape)
	print(X_train_shuff.shape, y_train_shuff.shape)
	print(type(X_test_shuff), type(y_test_shuff))
	print(type(X_train_shuff), type(y_train_shuff))
	print(' ')
	print(y_shuff_onehot_test.shape, y_shuff_onehot_train.shape)
	print(' ')
	print('Type of training data = ', type(training_data))
	print('Type of training labels = ', type(training_labels))
	print('Shape of training labels = ', training_labels.shape)
	print('Shape of training data = ', training_data.shape)

	"""Set model parameters"""
	learning_rate = 0.001
	batch_size = 128
	training_epochs = 200
	n_input = 1024  # Data input taps. 32 * 32 = 1024
	n_classes = 43  # Total classes
	layer_width = {
		'layer_1': 32,
		'layer_2': 64,
		'layer_3': 128,
		'fully_connected': 512
	}

#0 input 1 or 3 maps of 48x48 neurons
#1 convolutional 100 maps of 46x46 neurons 3x3
#2 max pooling 100 maps of 23x23 neurons 2x2

#3 convolutional 150 maps of 20x20 neurons 4x4
#4 max pooling 150 maps of 10x10 neurons 2x2

#5 convolutional 250 maps of 8x8 neurons 3x3
#6 max pooling 250 maps of 4x4 neurons 2x2
#7 fully connected 200 neurons
#8 fully connected 43 neurons

	# Store layers weight & bias
	weights = {
		'layer_1': tf.Variable(tf.truncated_normal(
			[3, 3, 1, layer_width['layer_1']])),
		'layer_2': tf.Variable(tf.truncated_normal(
			[4, 4, layer_width['layer_1'], layer_width['layer_2']])),
		'layer_3': tf.Variable(tf.truncated_normal(
			[3, 3, layer_width['layer_2'], layer_width['layer_3']])),
		'fully_connected': tf.Variable(tf.truncated_normal(
			[1024, layer_width['fully_connected']])),
		'out': tf.Variable(tf.truncated_normal(
			[layer_width['fully_connected'], n_classes]))
	}


# 	weights = {
# 		'layer_1': tf.Variable(tf.truncated_normal(
# 			[5, 5, 1, layer_width['layer_1']])),
# 		'layer_2': tf.Variable(tf.truncated_normal(
# 			[5, 5, layer_width['layer_1'], layer_width['layer_2']])),
# 		'layer_3': tf.Variable(tf.truncated_normal(
# 			[5, 5, layer_width['layer_2'], layer_width['layer_3']])),
# 		'fully_connected': tf.Variable(tf.truncated_normal(
# 			[1024, layer_width['fully_connected']])),
# 		'out': tf.Variable(tf.truncated_normal(
# 			[layer_width['fully_connected'], n_classes]))
# 	}

	biases = {
		'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
		'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
		'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
		'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
		'out': tf.Variable(tf.zeros(n_classes))
	}

	# tf Graph input
	x = tf.placeholder("float", [None, 32, 32, 1])
	y = tf.placeholder("float", [None, n_classes])

	logits = ml.conv_net(x, weights, biases)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
		.minimize(cost)

	# Initializing the variables
	init = tf.initialize_all_variables()
	saver = tf.train.Saver() # Object to save and restore variables.

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		"""Restore saved model for continued work"""
		if restore_model_for_continued_work == 1:
			print('Restoring model')
			saver = tf.train.import_meta_graph('model-checkpoint.meta')
			saver.restore(sess, 'model-checkpoint')
			all_vars = tf.trainable_variables()


		"""Train model"""
		if train_model == 1:
			for epoch in range(training_epochs):
				print('Starting epoch = ', epoch)
				start_time = time.time()
				total_batch = int(total_samples/batch_size)

				# Loop over all batches
				print('Processing batches. Not yet saving.')
				for i in range(total_batch):
					batch_x = training_data[i*batch_size:i*batch_size+batch_size,:,:]
					batch_y = training_labels[i*batch_size:i*batch_size+batch_size]

					# Run optimization op (backprop) and cost op (to get loss value)
					sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

				# Display logs per epoch step
				c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

				# Save model checkpoint
				print('Saving session')
				saver.save(sess, 'model-checkpoint')

				# Print info about the epoch
				elapsed_time = time.time() - start_time
				print('Time to process epoch (sec) = ', int(elapsed_time))
				print(' ')
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
			print("Optimization Finished!")

		"""Test model and report accuracy"""
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Accuracy:", ccuracy.eval({x: test_data, y: test_labels}))

if __name__ == "__main__":
    main()

