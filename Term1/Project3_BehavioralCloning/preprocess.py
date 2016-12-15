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
    Takes (x,y,3) numpy array and returns (x,y)
    """
    print('Making grayscale image')
    shape_image = image.shape
    print('Initial image shape = ', image.shape)
    ret_image = np.zeros((shape_image[0], shape_image[1]), dtype=np.float32)
    im1 = image[:,:,0]
    im2 = image[:,:,1]
    im3 = image[:,:,2]
    gray_im = im1/3.0 + im2/3.0 + im3/3.0
    ret_image[:,:] = gray_im
    print('Final image shape = ', ret_image.shape)
    return(ret_image)

def trim_image(image, start_row, stop_row):
    trimmed_image = image[start_row:stop_row,:,:] # Values chosen empirically. Trims out non-road data
    return trimmed_image

def show_trim_results(cen, lef, rig, cen_trim, lef_trim, rig_trim):
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

def show_grayscale(im):
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()

def normalize_image(image):
    print('Normalizing image')
    max = np.max(image)
    min = np.min(image)
    shape_image = image.shape
    ret_image = image / 255 - 0.5
    return ret_image

def combine_im(lef_proc, cen_proc, rig_proc):
    shape_im = lef_proc.shape
    ret_im = np.zeros((shape_im[0], shape_im[1], 3), dtype=np.float32)
    ret_im[:,:,0] = lef_proc
    ret_im[:,:,1] = cen_proc
    ret_im[:,:,2] = rig_proc
    return ret_im

def write_pickle_file(filename, y, X):
    print('Saving to pickle file')
    data_to_save = {'y': y,
                    'X': X,
                    }
    pickle.dump(data_to_save, open(filename, "wb" ))

def examine_data(index, y, X):
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

def load_pickle(file):
    print('Loading stats from file')
    data = pickle.load(open(file, "rb" ))
    y = data['y']
    X = data['X']
    return y, X

if __name__ == '__main__':
    # User-defined variables
    data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/'
    start_row = 60 # for trimming images
    stop_row = 140 # for trimming images
    pickle_filename = '/Users/blakejacquot/Desktop/temp2/proc_data.p'
    csv_path = os.path.join(data_path_root, 'driving_log.csv')
    csv_data = import_csv_data(csv_path)
    num_el = len(csv_data)
    print('Number of entries in csv file: %d' %num_el)

    # Make placeholder variables
    steer_ang_np = np.zeros(num_el, dtype=np.float32)
    #images = np.zeros((num_el, stop_row-start_row, 320, 3), dtype=np.float32)
    images = np.zeros((num_el, stop_row-start_row, 320, 1), dtype=np.float32)


    for i in range(num_el):
        print(i, ' of ', num_el)

        line = csv_data[i]

        # Get image paths and CSV values
        center_image_path = line[0]
        left_image_path = line[1]
        right_image_path = line[2]
        steering_angle = line[3]
        throttle = line[4]
        break_val = line[5]
        speed = line[6]

        # Set up labels
        steer_ang_float = float(steering_angle)
        steer_ang_np[i] = float(steering_angle)

        # Load images
        cen = load_image(center_image_path)
        #lef = load_image(left_image_path)
        #rig = load_image(right_image_path)

        # Trim images to only essential parts
        cen_proc = trim_image(cen, start_row, stop_row)
        #lef_proc = trim_image(lef, start_row, stop_row)
        #rig_proc = trim_image(rig, start_row, stop_row)
        #show_trim_results(cen, lef, rig, cen_proc, lef_proc, rig_proc)

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
        steer_ang_np = steer_ang_np.astype('float32')

        # Combine images together
        #final_im = combine_im(lef_proc, cen_proc, rig_proc)
        #final_im = combine_im(cen_proc)
        final_im = cen_proc


        images[i,:,:,0] = final_im
        images = images.astype('float32')

    #examine_data(0, steer_ang_np, images)
    write_pickle_file(pickle_filename, steer_ang_np, images)
    #y, X = load_pickle(pickle_filename)
    #examine_data(0, y, X)







