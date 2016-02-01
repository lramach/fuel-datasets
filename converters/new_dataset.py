import gzip
import os
import struct
import h5py
import numpy as np
from fuel.converters.base import fill_hdf5_file, check_exists

#@check_exists(required_files=ALL_FILES)
def convert_data(directory, output_file, dtype=None):
    """
    Convert new data to fuel HDF5 dataset.
    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_file : str
        Where to save the converted dataset.
    dtype : str, optional
        Either 'float32', 'float64', or 'bool'. Defaults to `None`,
        in which case images will be returned in their original
        unsigned byte format.
"""
    output_file = directory +"/new_dataset.hdf5"
    #print('output_file', output_file)
    #print('directory', directory)
    h5file = h5py.File(output_file, mode='w')

    train_file = os.path.join(directory, 'train_data.h5')
    train = h5py.File(train_file)
    x_train = np.array(train["chars"])
    y_train = np.array(train["target"])
    #Balancing the train set by selecting the first 18k responses from each class
    zeroes = np.transpose(np.array(np.where(y_train==0)))
    ones = np.transpose(np.array(np.where(y_train==1)))
    twos = np.transpose(np.array(np.where(y_train==2)))
    threes = np.transpose(np.array(np.where(y_train==3)))
    x_train = np.concatenate((x_train[zeroes[:18000, 0]], x_train[ones[:18000, 0]], x_train[twos[:18000, 0]], x_train[threes[:18000, 0]]), axis=0)
    y_train = np.concatenate((y_train[zeroes[:18000, 0]], y_train[ones[:18000, 0]], y_train[twos[:18000, 0]], y_train[threes[:18000, 0]]), axis=0)
    y_train = np.reshape(y_train, (len(y_train),1))
    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)
    #print('zeroes', len(y_train[y_train==0]))
    #print('ones', len(y_train[y_train==1]))
    #print('twos', len(y_train[y_train==2]))
    #print('threes', len(y_train[y_train==3]))
    valid_file = os.path.join(directory, 'valid_data.h5')
    valid = h5py.File(valid_file)
    x_valid = valid["chars"]
    y_valid = valid["target"]
    y_valid = np.reshape(y_valid, (len(y_valid), 1))
    test_file = os.path.join(directory, 'test_data.h5')
    test = h5py.File(test_file)
    x_test = test["chars"]
    y_test = test["target"]
    y_test = np.reshape(y_test, (len(y_test),1))
    features = x_train[:,:].astype('float32')
    targets = y_train[:].astype('uint8')
    train_features = features
    train_targets = targets
    print(train_targets.shape)
    features = x_valid[:,:].astype('float32')
    targets = y_valid[:].astype('uint8')
    valid_features = features
    valid_targets = targets
    print(valid_targets.shape)
    features = x_test[:,:].astype('float32')
    targets = y_test[:].astype('uint8')
    test_features = features
    test_targets = targets
    print(test_targets.shape)
    data = (('train', 'features', train_features),
            ('train', 'targets', train_targets),
            ('valid', 'features', valid_features),
            ('valid', 'targets', valid_targets),
            ('test', 'features', test_features),
            ('test', 'targets', test_targets))
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'feature'
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

def fill_subparser(subparser):
    """Sets up a subparser to convert the MNIST dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mnist` command.

    """
    subparser.add_argument(
        "--dtype", help="dtype to save to; by default, images will be " +
        "returned in their original unsigned byte format",
        choices=('float32', 'float64', 'bool'), type=str, default=None)
    subparser.set_defaults(func=convert_data)

