from sklearn.model_selection import KFold, cross_val_predict, train_test_split
import tensorflow.keras
from tensorflow.keras.layers import Input
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import numpy as np

def train_model_chunk(cnn_model, train_chunk_x, train_chunk_y, test_chunk_x, test_chunk_y):
    x_train, x_valid, y_train, y_valid = train_test_split(train_chunk_x, train_chunk_y, test_size=0.1)
    batch_size = 16
    epochs = 100
    model = Sequential()
    for layer in cnn_model.layers[:3]:
        layer.trainable = False
        model.add(layer)
        
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=0.73))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', name="conv2"))               
    model.add(MaxPooling2D(pool_size=(2, 2), name="max2"))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', name="den2"))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
 

    callbacks_list = [
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath='best_model_cnn1d.h5',
        monitor='val_loss', save_best_only=True)
    ]
    history1 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks_list,
          validation_data=(x_valid, y_valid))
    score = model.evaluate( test_chunk_x, test_chunk_y, verbose=0)
    
    return score[1], model

def self_learn(model, train_x, train_y, test_x, test_y, proportion_list, p_a):
    cnn_model1 = model
    results_list = []
    count_list = []
    accuracy_list = []
    i_count_list = []
    validation_accuracy_list = []

    proportion_list = [0.2,0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    p_a = [0.05,0.05,0.05,0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    


    for i in range(3): 
        print(i)
        sum_list = []
        count_l, class_count, train_X, train_Y, index_list, accuracy_l, i_count_l = add_pseudo_labels(cnn_model1, proportion_list, train_x, train_y, test_x, test_y )

        count_list.append(count_l)
        accuracy_list.append(accuracy_l)
        i_count_list.append(i_count_l)
        for (i1, i2) in zip(proportion_list,p_a):
            if ((i1+i2) >= 0.8):
                sum_list.append(0.8)
            else:
                sum_list.append(i1+i2)
    
        proportion_list = sum_list
        cnn_model1 = train_model(train_X, train_Y, test_x, test_y, train_x, train_y, i)
        accu = print_accuracy(cnn_model1, test_x, test_y)
        results_list.append(accu)
        print("the test accuracy for this training round:")
        print(accu)
    return results_list


def train_model(train_chunk_x, train_chunk_y, x_test, y_test,x_train, y_train, i):
    source_labels = np.argmax(y_train, axis=1)
    target_labels = np.argmax(y_test, axis=1)
    com_chunk_x = np.vstack((x_train, x_test))
    com_chunk_y = np.vstack((y_train, y_test))
    x_train, x_valid, y_train, y_valid = train_test_split(train_chunk_x, train_chunk_y, test_size=0.1)
    model = Sequential()
    for layer in cnn_model.layers[:3]:
        layer.trainable = False
        model.add(layer)
    
    model.add(Conv2D(128, (3, 3), activation='relu', name="conv1"))
    model.add(BatchNormalization(momentum=0.73))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxl1"))
    model.add(Conv2D(256, (3, 3), activation='relu', name="conv2"))
    # cnn_model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2), name="max2"))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', name="den2"))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
    batch_size = 16

    callbacks_list = [
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath='best_model_cnn1d.h5',
        monitor='val_loss', save_best_only=True)
    ]
    history1 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=50,
          verbose=2,
          callbacks=callbacks_list,
          validation_data=(x_valid, y_valid))
    
    return model

def add_pseudo_labels(cnn_model1, proportion_list, train_x, train_y, test_x, test_y):
    pred_test = predict_test(cnn_model1, test_x)
    class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    temp_list = []
    count_list = []
    accuracy_list = []
    index_count_list = []
#     count_1 = 0
    for i in range(len(class_list)):
        temp_index, accuracy, index_count = get_pseudo_index(class_list[i], pred_test, proportion_list[i], test_y)
        if (type(temp_index) != np.ndarray):
            count_list.append(0)
        else:
            count_list.append(len(temp_index))
        accuracy_list.append(accuracy)
        index_count_list.append(index_count)
        if (i ==0):
            temp_list = temp_index
        else:
            if (type(temp_index) == np.ndarray):
                temp_list = np.concatenate((temp_list,temp_index))
            
    temp_list = np.array(temp_list).flatten()
    train_X = np.vstack((train_x, test_x[temp_list]))
    train_Y = np.vstack((train_y, pred_test[temp_list]))
    index_class = [index for index,value in enumerate(np.argmax(pred_test, axis=1)) if value == class_list[0]]
    return np.array(count_list), len(index_class), train_X, train_Y, temp_list, np.array(accuracy_list), np.array(index_count_list)

    
def predict_test(cnn_model1, test_x):
    pred_test = cnn_model1.predict(test_x)
    return pred_test

def get_pseudo_index(class_num, y_test_pred, proportion, test_y):
    index_class = [index for index,value in enumerate(np.argmax(y_test_pred, axis=1)) if value == class_num]
    index_class = np.array(index_class)
    if (proportion > 0):
        index_class_array = np.argpartition(y_test_pred[index_class][:,class_num], -int(index_class.shape[0] * proportion))[-int(index_class.shape[0] * proportion):]
        accuracy = accuracy_score(np.argmax(y_test_pred[index_class][index_class_array], axis=1), np.argmax(test_y[index_class][index_class_array],axis=1))
        return index_class[index_class_array], accuracy, index_class.shape[0]
    else:
        return 0, 0, index_class.shape[0]


def print_accuracy(cnn_model1, test_x, test_y):
    a = cnn_model1.predict(test_x)
    acc = accuracy_score(np.argmax(test_y, axis=1), np.argmax(a, axis=1))
    return acc