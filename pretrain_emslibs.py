from tensorflow.keras.models import model_from_json
import h5py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_model(json_filepath, weights_filepath):
    # load json and create model
    with open(json_filepath, 'r') as json_file:
        loaded_model_json = json_file.read()
    cnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    cnn_model.load_weights(weights_filepath)
    print("Loaded model from disk")
    return cnn_model

def get_file_path(file, filename):
    return os.path.dirname(file) +'/LIBS/' + filename

def get_data(file_name, spectraCount):
    f = h5py.File(file_name, 'r')
    wavelengths = f["Wavelengths"]
    wavelengths = wavelengths[list(wavelengths.keys())[0]].value
    data = None
    for sample in list(f["Spectra"].keys()):
        tempData = f["Spectra"][sample].value
        tempData = tempData[:,0:spectraCount]
        if data is None:
            data = tempData.transpose()
        else:
            data = np.append(data, tempData.transpose(), axis = 0)
    return data, wavelengths

def get_class(file_name, spectraCount):
    f = h5py.File(file_name, 'r')
    trainClass = f["Class"]["1"].value
    tempClass = None
    for i in range(0,50000,500):
        if i == 0:
            tempClass = trainClass[0:spectraCount]
        else:
            tempClass = np.append(tempClass, trainClass[i:(i+spectraCount)])
    return tempClass

def get_test_data(file_name):
    f = h5py.File(file_name, 'r')
    testData = None
    for sample in list(f["UNKNOWN"].keys()):
        tempData = f["UNKNOWN"][sample].value
        if testData is None:
            testData = tempData.transpose()
        else:
            testData = np.append(testData, tempData.transpose(), axis = 0)
    return testData

def get_test_class():
    df=pd.read_csv('test_labels.csv', sep=',',header=None)
    return df.values

def normalize_data(trainData, testData):
    minmax_scale = MinMaxScaler(feature_range=(0, 1)).fit(trainData.T)
    trainData_norm = minmax_scale.transform(trainData.T).T
    testData_norm = minmax_scale.transform(testData.T).T
    return trainData_norm, testData_norm

def reshape_data(trainData_norm, testData_norm, trainClass, testClass):
    train_x = trainData_norm.reshape(trainData_norm.shape[0],177,226,1)
    test_x = testData_norm.reshape(testData_norm.shape[0],177,226,1)
    train_y = trainClass.reshape(trainClass.shape[0],1)
    test_y = testClass.reshape(testClass.shape[0],1)
    return train_x, test_x, train_y, test_y

def split_data(train_x, trainClass):
    return train_test_split(train_x, trainClass, test_size=0.2, random_state=42)

def main():
    file = '/home/huang/LIBS'
    spectraCount = 200
    train_file_name = get_file_path(file, 'train.h5')
    test_file_name = get_file_path(file, 'test.h5')
    trainData, wavelengths = get_data(train_file_name, spectraCount)
    trainClass = get_class(train_file_name, spectraCount)
    testData = get_test_data(test_file_name)
    testClass = get_test_class()
    trainData_norm, testData_norm = normalize_data(trainData, testData)
    train_x, test_x, train_y, test_y = reshape_data(trainData_norm, testData_norm, trainClass, testClass)
    x_train, x_valid, y_train, y_valid = split_data(train_x, trainClass)
    train_Y_one_hot = to_categorical(trainClass)
    test_Y_one_hot = to_categorical(testClass,13)

if __name__ == "__main__":
    main()