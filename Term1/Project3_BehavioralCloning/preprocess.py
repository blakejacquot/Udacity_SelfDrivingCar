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
import PIL

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
    pass

def trim_image(image):
    trimmed_image = image[60:-20,:,:] # Values chosen empirically. Trims out non-road data
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


if __name__ == '__main__':
    # User-defined variables
    data_path_root = '/Users/blakejacquot/Desktop/temp2/DrivingSimulator/'

    csv_path = os.path.join(data_path_root, 'driving_log.csv')
    csv_data = import_csv_data(csv_path)
    num_el = len(csv_data)
    print('Number of entries in csv file: %d' %num_el)
    line = csv_data[3000]


    center_image_path = line[0]
    left_image_path = line[1]
    right_image_path = line[2]
    steering_angle = line[3]
    throttle = line[4]
    break_val = line[5]
    speed = line[6]

    cen = load_image(center_image_path)
    lef = load_image(left_image_path)
    rig = load_image(right_image_path)

    print(cen.shape)
    print(lef.shape)
    print(rig.shape)

    cen_trim = trim_image(cen)
    lef_trim = trim_image(lef)
    rig_trim = trim_image(rig)


    show_trim_results(cen, lef, rig, cen_trim, lef_trim, rig_trim)

    plt.show()

