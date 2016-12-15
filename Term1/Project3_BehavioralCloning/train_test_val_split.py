import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

def load_pickle(file):
    print('Loading stats from file')
    data = pickle.load(open(file, "rb" ))
    y = data['y']
    X = data['X']
    return y, X

def examine_data(index, y, X):
    print('Label = ', y[index], type(y[index]))
    print(np.max(X), np.min(X))
    image = X[index,:,:,:]
    print('Shape of image = ', image.shape)
    print(np.min(image), np.max(image))
    lef = image[:,:,0]
    #cen = image[:,:,1]
    #rig = image[:,:,2]
    cen = lef
    rig = lef
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

def write_pickle_file(filename, y, X):
    print('Saving to pickle file')
    data_to_save = {'y': y,
                    'X': X,
                    }
    pickle.dump(data_to_save, open(filename, "wb" ))

if __name__ == '__main__':

    # User-defined variables
    pickle_filename = '/Users/blakejacquot/Desktop/temp2/proc_data.p' # Initial data
    train_path = '/Users/blakejacquot/Desktop/temp2/data_train.p'
    test_path = '/Users/blakejacquot/Desktop/temp2/data_test.p'
    val_path = '/Users/blakejacquot/Desktop/temp2/data_val.p'

    y, X = load_pickle(pickle_filename)
    print(type(y), type(X))
    print(y.shape, X.shape)
    examine_data(3000, y, X)

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
