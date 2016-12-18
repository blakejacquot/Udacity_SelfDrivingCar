"""
Import and preprocess image and csv data.

Each sample in dataset has left, center, and right images and a single line from csv.

CSV file structure:
Center image, left image, right image, steering angle, throttle, break, speed

Split data into train, test, and validation sets.

Save results by pickle
"""

import csv
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def import_csv_data(cvs_path):
    """ Import CSV data from file.
    """
    print('Importing CSV data')
    data = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            data.append(line)
    return data

def load_image(im_path):
    """ Load image from file path
    """
    im_path_stripped = im_path.strip() # Removes white space if present
    curr_image = plt.imread(im_path_stripped)
    return curr_image

def grayscale(image):
    """
    Takes (x,y,3) RGB numpy array and returns grayscale (x,y)
    """
    shape_image = image.shape
    #print('Initial image shape = ', image.shape)
    ret_image = np.zeros((shape_image[0], shape_image[1]))
    im1 = image[:, :, 0]
    im2 = image[:, :, 1]
    im3 = image[:, :, 2]
    gray_im = im1/3.0 + im2/3.0 + im3/3.0
    ret_image[:, :] = gray_im
    #print('Final image shape = ', ret_image.shape)
    return ret_image

def show_grayscale(curr_image):
    """ Make RGB image into grayscale.
    """
    plt.figure()
    plt.imshow(curr_image, cmap='gray')
    plt.show()

def trim_image(image, start_row, stop_row):
    """ Trim all non-road data (e.g. trees, scenery).
    """
    trimmed_image = image[start_row:stop_row, :, :] # Trim non-road data
    return trimmed_image

def show_trim_results(cen, lef, rig, cen_trim, lef_trim, rig_trim):
    """ Examine RBG images after trimming rows
    """
    #plt.figure
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
    ret_image = image / 255 - 0.5
    return ret_image

def combine_im(lef_proc, cen_proc, rig_proc):
    """ Vestigial. May delete since game does not give left, right images.
    """
    shape_im = lef_proc.shape
    ret_im = np.zeros((shape_im[0], shape_im[1], 3))
    ret_im[:, :, 0] = lef_proc
    ret_im[:, :, 1] = cen_proc
    ret_im[:, :, 2] = rig_proc
    return ret_im

def write_pickle_file(filename, y_labels, x_images):
    """ Write X, y (Image, label) data to pickle file.
    """
    print('Saving to pickle file')
    data_to_save = {'y': y_labels,
                    'X': x_images}
    pickle.dump(data_to_save, open(filename, "wb"))

def load_pickle(file):
    """ Read X, y (Image, label) data to pickle file.
    """
    print('Loading stats from file')
    data = pickle.load(open(file, "rb"))
    y = data['y']
    X = data['X']
    return y, X

def examine_data(index, y_labels, x_images):
    """ Display details of chosen label and image data
    """
    print('Label = ', y_labels[index], type(y_labels[index]))
    print(np.max(x_images), np.min(x_images))
    image = x_images[index, :, :, :]
    print('Shape of image = ', image.shape)
    print(np.min(image), np.max(image))
    lef = image[:, :, 0]
    cen = image[:, :, 1]
    rig = image[:, :, 2]
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
    DATA_PATH_ROOT = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset2/'
    DATA_PATH_ROOT = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset3/'
    DATA_PATH_ROOT = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/BCJ_dataset4/'

    START_ROW = 60 # for trimming images
    STOP_ROW = 140 # for trimming images

    # Set up paths and some variables
    PICKLE_FILEPATH = os.path.join(DATA_PATH_ROOT, 'proc_data.p')
    CSV_PATH = os.path.join(DATA_PATH_ROOT, 'driving_log.csv')
    CSV_DATA = import_csv_data(CSV_PATH)
    NUM_EL = len(CSV_DATA)
    print('Number of entries in csv file: %d' %NUM_EL)

    # Make placeholder variables
    STEER_ANG_NP = np.zeros(NUM_EL)
    images = np.zeros((NUM_EL, STOP_ROW-START_ROW, 320, 1))


    for i in range(1, NUM_EL):
        print('Processing ', i, ' of ', NUM_EL)

        # Get image paths and CSV values
        curr_line = CSV_DATA[i]
        center_image_path = curr_line[0]
        left_image_path = curr_line[1]
        right_image_path = curr_line[2]
        steering_angle = curr_line[3]
        throttle = curr_line[4]
        break_val = curr_line[5]
        speed = curr_line[6]

        # Set up labels
        steer_ang_float = float(steering_angle)
        STEER_ANG_NP[i] = float(steering_angle)

        # Load images
        split_path = os.path.split(center_image_path)
        filename = split_path[1]
        cen_full_path = os.path.join(DATA_PATH_ROOT, 'IMG', filename)
        cen = load_image(cen_full_path)
        #lef = load_image(left_image_path)
        #rig = load_image(right_image_path)

        # Trim images to only essential parts
        cen_proc = trim_image(cen, START_ROW, STOP_ROW)
        #lef_proc = trim_image(lef, STOP_ROW, STOP_ROW)
        #rig_proc = trim_image(rig, STOP_ROW, STOP_ROW)
        #show_trim_results(cen, lef, rig, cen_proc, lef_proc, rig_proc) # For troubleshooting


        # Grayscale the image
        cen_proc = grayscale(cen_proc)
        #lef_proc = grayscale(lef_proc)
        #rig_proc = grayscale(rig_proc)
        #show_grayscale(cen_proc)

        # Normalize the image
        cen_proc = normalize_image(cen_proc)

        #lef_proc = normalize_image(lef_proc)
        #rig_proc = normalize_image(rig_proc)

        # Ensure data is float32 (Tensorflow expects this)
        cen_proc = cen_proc.astype('float32')
        #lef_proc = lef_proc.astype('float32')
        #rig_proc = rig_proc.astype('float32')
        STEER_ANG_NP = STEER_ANG_NP.astype('float32')

        # Combine images together
        #final_im = combine_im(lef_proc, cen_proc, rig_proc)
        #final_im = combine_im(cen_proc)
        final_im = cen_proc

        images[i, :, :, 0] = final_im
        images = images.astype('float32')

    write_pickle_file(PICKLE_FILEPATH, STEER_ANG_NP, images)

    #y, X = load_pickle(pickle_filename) # For troubleshooting
    #examine_data(0, y, X) # For trouble shooting

    # Do train-test-split
    PICKLE_PATH = os.path.join(DATA_PATH_ROOT, 'proc_data.p')
    TRAIN_PATH = os.path.join(DATA_PATH_ROOT, 'data_train.p')
    TEST_PATH = os.path.join(DATA_PATH_ROOT, 'data_test.p')
    VAL_PATH = os.path.join(DATA_PATH_ROOT, 'data_val.p')
    Y_LABELS, X_IMAGES = load_pickle(PICKLE_PATH)
    #examine_data(3000, y, X)
    y_shape = Y_LABELS.shape
    X_shape = X_IMAGES.shape
    NUM_EL = y_shape[0]
    print(y_shape, X_shape, NUM_EL)

    # Want training, test split to be 80%, 20% of total data
    # Of the remaining training data, want training, validation split to be 80%, 20%
    # Make train, test split
    [train_data, test_data, train_labels, test_labels] = train_test_split(X_IMAGES, Y_LABELS,
                                                                          test_size=0.20,
                                                                          random_state=101)
    # Make train, val split
    [train_data, val_data, train_labels, val_labels] = train_test_split(train_data,
                                                                        train_labels,
                                                                        test_size=0.20,
                                                                        random_state=101)
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
    write_pickle_file(TRAIN_PATH, train_labels, train_data)
    write_pickle_file(TEST_PATH, test_labels, test_data)
    write_pickle_file(VAL_PATH, val_labels, val_data)









