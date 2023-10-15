# fn for the dateset (convert jpg to hdf5) and to load the data set.
import h5py
import numpy as np


def convert_file(input_dir, filename, output_dir):
    filepath = input_dir + '/' + filename
    fin = open(filepath, 'rb')
    binary_data = fin.read()
    new_filepath = output_dir + '/' + filename[:-4] + '.hdf5'
    f = h5py.File(new_filepath)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = f.create_dataset('binary_data', (100, ), dtype=dt)
    dset[0] = np.fromstring(binary_data, dtype='uint8')


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels

    return X_train, y_train, X_test, y_test
