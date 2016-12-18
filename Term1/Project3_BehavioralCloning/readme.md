# Overview
This is the submission for the Udacity self-driving car class, Term 1, Project 3 (Transfer
Learning)

## Background information
Example use:
python drive.py drive.py model.json

CSV file structure:
Center image, left image, right image, steering angle, throttle, break, speed

Collection of training data is organized into IMG folder (containing all images) and
driving_log.csv.

## Modules
### preprocess.py
Preprocesses image and csv data for use in training.

### model.py
Use preprocessed image and csv data to train a model

### drive.py
Set up to run with the simulator.

## Preprocessing
The file 'preprocess.py' executes a pipelined approach after reading in CSV file and
finding the images. First, images are cropped to remove all non-road data. This mostly
involves cropping out the upper portion of the image. Second, the images are converted from
RGB to grayscale. This helps with speed of training and prevents overfitting to the colors
of a scene. Third, the images are normalized from 0/255 to -1.0/+1.0.

## Model Architecture and Training Strategy

### Model architecture
The model more or less follows the NVidia architecture. It has 5 convnets followed by
flattening followed by 3 fully-connected layers. The final output is a single value
representing the steering angle. The number of outputs, kernel sizes, and strides were
chosen to match the Nvidia architecture. The only exception is the final convnet where
kernel size was chosen to be 3x1 vs 3x3. This is because the output of the prior layer
did not allow for 3x3, presumably because I'm starting with a different image size.

Perhaps different than Nvidia architecture, I chose a 20% dropout rate and Relu activation
after each convnet. Dropout helps prevent overfitting. Activation breaks linearity in
the network and allows learning more complex functions than linear regression.

### Train/val/test splits
Just before ending, 'preprocess.py' executes the train/test/val split on a dataset. The
split is first a 80%/20% train/test split. Of the remaining training data, there is another
80%/20% train/val split.

### Model parameters
Optimizer: Use the Adam optimizer.

Learning rate: Start with default 0.001 rate from Adam optimizer and change to 0.0001 as
we fine-tune the model.

Batch size: Larger batch size is more efficient from parallel computing perspective, but
it reduces sample variance and increases risk of overfitting.

Epochs: Monitor loss calculation to ensure it decreases over epoch. This indicates, we
haven't reached the minima. In my approach of using multiple datasets, the loss always
decreased because there's so much variance in the data. I used the number of epochs to
over/underweight a given dataset.

### Datasets
'BCJ Udacity': 5100 training samples, unknown number of laps or driving style
'BCJ_dataset1': 1900 training samples, 2 laps of normal driving
'BCJ_dataset2': 2000 training samples, 2 laps of slightly drunk driving
'BCJ_dataset3': 4200 training samples, 3 laps of drunk driving
'BCJ_dataset4': 500 training samples, special situations

In the above datasets, 'drunk driving' represents careening towards edges of track and
recovering. This helps the car recover from extreme situations. The special situation
dataset represents extra samples where the track differs in character and provides extra
training here.