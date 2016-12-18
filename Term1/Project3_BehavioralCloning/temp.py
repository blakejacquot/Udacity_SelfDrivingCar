"""
Import and preprocess image and csv data.

Each sample in dataset has left, center, and right images and a single line from csv.

CSV file structure:
Center image, left image, right image, steering angle, throttle, break, speed

Split data into train, test, and validation sets.

Save results by pickle
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
from sklearn.model_selection import train_test_split


def import_csv_data(cvs_path):
    print('Importing CSV data')
    data = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            data.append(line)
    return data

def load_image(im_path):
    im_path_stripped = im_path.strip() # Removes white space if present
    im = plt.imread(im_path_stripped)
    return im

def grayscale(image):
    """
    Takes (x,y,3) RGB numpy array and returns grayscale (x,y)
    """
    shape_image = image.shape
    #print('Initial image shape = ', image.shape)
    ret_image = np.zeros((shape_image[0], shape_image[1]), dtype=np.float32)
    im1 = image[:,:,0]
    im2 = image[:,:,1]
    im3 = image[:,:,2]
    gray_im = im1/3.0 + im2/3.0 + im3/3.0
    ret_image[:,:] = gray_im
    #print('Final image shape = ', ret_image.shape)
    return(ret_image)

def show_grayscale(im):
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()

def trim_image(image, start_row, stop_row):
    """ Trim all non-road data (e.g. trees, scenery).
    """
    trimmed_image = image[start_row:stop_row,:,:] # Values chosen empirically. Trims out non-road data
    return trimmed_image

def show_trim_results(cen, lef, rig, cen_trim, lef_trim, rig_trim):
    """ Examine RBG images after trimming rows
    """
    plt.figure
    plt.subplot(231)
    plt.imshow(lef)
    plt.title('Left pre-trim')
    plt.subplot(232)
    plt.imshow(cen)
    plt.title('Center pre-trim')
    plt.subplot(233)
    plt.imshow(rig)
    plt.title('Right pre-trim')
    plt.subplot(234)
    plt.imshow(lef_trim)
    plt.title('Left post-trim')
    plt.subplot(235)
    plt.imshow(cen_trim)
    plt.title('Center post-trim')
    plt.subplot(236)
    plt.imshow(rig_trim)
    plt.title('Right post-trim')
    plt.show()

def normalize_image(image):
    """ Change from 0-255 space to -1.0 to 1.0 space
    """
    max = np.max(image)
    min = np.min(image)
    shape_image = image.shape
    ret_image = image / 255 - 0.5
    return ret_image

def combine_im(lef_proc, cen_proc, rig_proc):
    """ Vestigial. May delete since game does not give left, right images.
    """
    shape_im = lef_proc.shape
    ret_im = np.zeros((shape_im[0], shape_im[1], 3), dtype=np.float32)
    ret_im[:,:,0] = lef_proc
    ret_im[:,:,1] = cen_proc
    ret_im[:,:,2] = rig_proc
    return ret_im

def write_pickle_file(filename, y, X):
    """ Write X, y (Image, label) data to pickle file.
    """
    print('Saving to pickle file')
    data_to_save = {'y': y,
                    'X': X,
                    }
    pickle.dump(data_to_save, open(filename, "wb" ))

def load_pickle(file):
    """ Read X, y (Image, label) data to pickle file.
    """
    print('Loading stats from file')
    data = pickle.load(open(file, "rb" ))
    y = data['y']
    X = data['X']
    return y, X

def examine_data(index, y, X):
    """ Display details of chosen label and image data
    """
    print('Label = ', y[index], type(y[index]))
    print(np.max(X), np.min(X))
    image = X[index,:,:,:]
    print('Shape of image = ', image.shape)
    print(np.min(image), np.max(image))
    lef = image[:,:,0]
    cen = image[:,:,1]
    rig = image[:,:,2]
    print(lef.shape, cen.shape, rig.shape)
    min = np.min(lef)
    max = np.max(lef)
    print(min, max)
    plt.figure
    plt.subplot(131)
    plt.imshow(lef, cmap='gray')
    plt.title('Left')
    plt.subplot(132)
    plt.imshow(cen, cmap='gray')
    plt.title('Center')
    plt.subplot(133)
    plt.imshow(rig, cmap='gray')
    plt.title('Right')
    plt.show()

if __name__ == '__main__':
    # User-defined variables
    #data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/Data_from_Udacity/'
    #data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset1/'
    data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset2/'
    data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset3/'


    """Do train-test-split"""
    pickle_path = os.path.join(data_path_root, 'proc_data.p')
    train_path = os.path.join(data_path_root, 'data_train.p')
    test_path = os.path.join(data_path_root, 'data_test.p')
    val_path = os.path.join(data_path_root, 'data_val.p')
    y, X = load_pickle(pickle_path)
    print(type(y), type(X))
    print(y.shape, X.shape)
    #examine_data(3000, y, X)
    y_shape = y.shape
    X_shape = X.shape
    num_el = y_shape[0]
    print(y_shape, X_shape, num_el)
    # Want training, test split to be 80%, 20% of total data
    # Of the remaining training data, want training, validation split to be 80%, 20%
    [train_data, test_data, train_labels, test_labels] = train_test_split(X, y, test_size=0.20, random_state=101)
    [train_data, val_data, train_labels, val_labels] = train_test_split(train_data, train_labels, test_size=0.20, random_state=101)
    #"""Report General statistics on training, testing, validation data sets"""
    # Want training, test split to be 80%, 20% of total data
    # Of the remaining training data, want training, validation split to be 80%, 20%
    numel_train = train_labels.shape[0]
    numel_test = test_labels.shape[0]
    numel_validation = val_labels.shape[0]
    total_samples = numel_train + numel_test + numel_validation
    print('Total samples = ', total_samples)
    print('Testing as percentage of whole = %f' % (numel_test/total_samples))
    print('Training as percentage of whole = %f' % (numel_train/total_samples))
    print('Validation as percentage of whole = %f' % (numel_validation/total_samples))
    write_pickle_file(train_path, train_labels, train_data)
    write_pickle_file(test_path, test_labels, test_data)
    write_pickle_file(val_path, val_labels, val_data)









