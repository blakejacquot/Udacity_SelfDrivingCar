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


"""Pre processing helper functions"""
def make_gaussian_blur(x, kernel_size):
    x_shape = x.shape
    print(x_shape)
    num_el = x_shape[0]
    ret_images = np.ones((x_shape[0],x_shape[1],x_shape[2]))
    print(ret_images.shape)
    for i in range(num_el):
        curr_im = x[i][:][:][:]
        ret_images[i][:][:] = gaussian_blur(curr_im, kernel_size)
    return ret_images

def crop_to_ROI(x, vertices):
    x_shape = x.shape
    print(x_shape)
    num_el = x_shape[0]
    ret_images = np.ones((x_shape[0],x_shape[1],x_shape[2]))
    print(ret_images.shape)
    for i in range(num_el):
        curr_im = x[i][:][:][:]
        ret_images[i][:][:] = get_ROI(curr_im, vertices)
    return ret_images

def normalize(x):
    x_shape = x.shape
    print(x_shape)
    num_el = x_shape[0]
    ret_images = np.ones((x_shape[0],x_shape[1],x_shape[2]))
    print(ret_images.shape)
    for i in range(num_el):
        curr_im = x[i][:][:][:]
        empty_im = np.ones((x_shape[1],x_shape[2]))
        #print('Normalizing image')
        #print(np.ndarray.max(curr_im), np.ndarray.min(curr_im))
        #proc_im = cv2.normalize(curr_im, empty_im, 0,255,cv2.NORM_MINMAX)
        proc_im = cv2.normalize(curr_im, empty_im, -127,128,cv2.NORM_MINMAX)
        #print(np.ndarray.max(proc_im), np.ndarray.min(proc_im))
        ret_images[i][:][:] = proc_im
    return ret_images


"""Helper functions from Project 1"""
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def get_ROI(img, vertices):
    pass

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    Args:
      img:
      vertices:

    Returns:

    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns
      line_img: Image with hough lines drawn.
      lines: Hough lines from the transform of form x1,y1,x2,y2.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    Args:
      img: Output of the hough_lines(), An image with lines drawn on it.
           Should be a blank image (all black) with lines drawn on it.
      initial_img: image before any processing.
      α: TBD
      β: TBD
      λ: TBD

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


"""Helper Functions"""


def make_grayscale(x):
    x_shape = x.shape
    print(x_shape)
    num_el = x_shape[0]
    ret_images = np.ones((x_shape[0],x_shape[1],x_shape[2]))
    print(ret_images.shape)
    for i in range(num_el):
        curr_im = x[i][:][:][:]
        ret_images[i][:][:] = grayscale(curr_im)
    return ret_images

# def make_grayscale(x):
#     x_shape = x.shape
#     num_el = x_shape[0]
#     ret_images = np.ones((x_shape[0],x_shape[1],x_shape[2]))
#     for i in range(num_el):
#         curr_im = x[i][:][:][:]
#         r = curr_im[:,:,0]
#         b = curr_im[:,:,1]
#         g = curr_im[:,:,2]
#         curr_im_gray = (r) / 3
#         ret_images[i][:][:] = curr_im_gray
#     return ret_images




# """Helper functions for data categorization, preprocessing, and exploration"""
# def make_class_dict(y):
#     class_dict = {}
#     num_el = len(y)
#     for i in range(num_el):
#         curr_class = y[i]
#         if curr_class not in class_dict.keys():
#             class_dict[curr_class] = [i]
#         else:
#             pos_index = class_dict[curr_class]
#             pos_index.append(i)
#             class_dict[curr_class] = pos_index
#     return class_dict

# def plot_random(X, class_dict):
#     for curr_class in class_dict.keys():
#         pos_index = class_dict[curr_class]
#         len_index = len(pos_index)
#         i1 = random.randrange(len_index)
#         i2 = random.randrange(len_index)
#         i3 = random.randrange(len_index)
#         print('Current class = ' + str(curr_class))
#         index1 = pos_index[i1]
#         index2 = pos_index[i2]
#         index3 = pos_index[i3]
#         im1 = X[index1][:][:][:]
#         im2 = X[index2][:][:][:]
#         im3 = X[index3][:][:][:]
#         plt.figure()
#         plt.subplot(131)
#         plt.imshow(im1, cmap='Greys_r')
#         plt.subplot(132)
#         plt.imshow(im2, cmap='Greys_r')
#         plt.subplot(133)
#         plt.imshow(im3, cmap='Greys_r')
#         plt.show()
#         mean_im1, max_im1 = np.mean(im1), np.max(im1)
#         mean_im2, max_im2 = np.mean(im2), np.max(im2)
#         mean_im3, max_im3 = np.mean(im3), np.max(im3)
#         print('Mean of im1,2,3 = ', mean_im1, mean_im2, mean_im3)
#         print('Max of im1,2,3 = ', max_im1, max_im2, max_im3)
#     plt.close("all")


